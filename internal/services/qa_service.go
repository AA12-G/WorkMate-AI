package services

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"WorkMate-AI/config"
	"WorkMate-AI/internal/models"
	"WorkMate-AI/pkg/vectorstore"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type QAService struct {
	llm    llms.LLM
	qdrant *vectorstore.QdrantStore
	config *config.Config
}

// 创建一个自定义的 embedder
type LocalEmbedder struct {
	llm llms.LLM
}

// EmbedDocuments 实现 embeddings.Embedder 接口
func (e *LocalEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	// 使用随机数生成器
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		vector := make([]float32, 3072)
		for j := range vector {
			vector[j] = rand.Float32() // 添加随机值
		}
		embeddings[i] = vector
	}
	return embeddings, nil
}

// EmbedQuery 实现 embeddings.Embedder 接口
func (e *LocalEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	vector := make([]float32, 3072)
	for i := range vector {
		vector[i] = rand.Float32() // 添加随机值
	}
	return vector, nil
}

func NewQAService(cfg *config.Config) (*QAService, error) {
	var llmInstance llms.LLM
	var err error

	// 使用 Ollama 作为 LLM
	llmInstance, err = ollama.New(
		ollama.WithModel(cfg.OllamaModel),
	)
	if err != nil {
		return nil, err
	}

	// 使用本地 embedder
	embedder := &LocalEmbedder{llm: llmInstance}

	// 初始化 Qdrant 存储
	qdrantStore, err := vectorstore.NewQdrantStore(
		context.Background(),
		cfg.QdrantGRPCURL,
		cfg.QdrantHTTPURL,
		cfg.CollectionName,
		embedder,
	)
	if err != nil {
		return nil, err
	}

	return &QAService{
		llm:    llmInstance,
		qdrant: qdrantStore,
		config: cfg,
	}, nil
}

// 修改辅助函数来处理 Qdrant URL
func getQdrantHostAndPort(url string) (string, int) {
	// 移除 http:// 或 https:// 前缀
	url = strings.TrimPrefix(url, "http://")
	url = strings.TrimPrefix(url, "https://")

	// 如果 URL 包含端口号，分离主机名和端口
	if strings.Contains(url, ":") {
		parts := strings.Split(url, ":")
		if len(parts) == 2 {
			// 只取第一个冒号前后的部分
			return parts[0], 6333
		}
	}

	// 如果没有端口号或格式不正确，返回默认值
	return url, 6333
}

// ProcessFile 处理上传的文件
func (s *QAService) ProcessFile(ctx context.Context, filePath string) error {
	// 打开文件
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	fileInfo, err := f.Stat()
	if err != nil {
		return err
	}

	// 根据文件类型选择合适的加载器
	var loader documentloaders.Loader
	ext := strings.ToLower(path.Ext(filePath))

	switch ext {
	case ".pdf":
		loader = documentloaders.NewPDF(f, fileInfo.Size())
	case ".txt", ".md":
		loader = documentloaders.NewText(f)
	case ".csv":
		loader = documentloaders.NewCSV(f)
	case ".doc", ".docx":
		return fmt.Errorf("暂不支持 Word 文档")
	default:
		return fmt.Errorf("不支持的文件类型: %s", ext)
	}

	// 分割文档
	chunkDocuments, err := loader.LoadAndSplit(ctx, textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(2000),
		textsplitter.WithChunkOverlap(200),
	))
	if err != nil {
		return err
	}

	// 添加文档到向量存储
	_, err = s.qdrant.GetStore().AddDocuments(ctx, chunkDocuments)
	if err != nil {
		return err
	}

	// 保存文档元数据
	doc := models.Document{
		ID:          uuid.New().String(),
		Name:        filepath.Base(filePath),
		Type:        filepath.Ext(filePath),
		Size:        fileInfo.Size(),
		UploadTime:  time.Now(),
		Description: "",
	}

	// 创建 gRPC 客户端
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	// 首先创建集合
	collectionsClient := qdrant.NewCollectionsClient(conn)
	_, err = collectionsClient.Create(ctx, &qdrant.CreateCollection{
		CollectionName: s.config.DocumentMetadataCollection,
		VectorsConfig: &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: &qdrant.VectorParams{
					Size:     3072,
					Distance: qdrant.Distance_Cosine,
				},
			},
		},
	})
	if err != nil && !strings.Contains(err.Error(), "already exists") {
		return fmt.Errorf("创建集合失败: %v", err)
	}

	// 使用 Points 客户端
	pointsClient := qdrant.NewPointsClient(conn)

	vector := make([]float32, 3072)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	payload := make(map[string]*qdrant.Value)
	payload["name"] = &qdrant.Value{
		Kind: &qdrant.Value_StringValue{
			StringValue: doc.Name,
		},
	}
	payload["type"] = &qdrant.Value{
		Kind: &qdrant.Value_StringValue{
			StringValue: doc.Type,
		},
	}
	payload["size"] = &qdrant.Value{
		Kind: &qdrant.Value_IntegerValue{
			IntegerValue: doc.Size,
		},
	}
	payload["upload_time"] = &qdrant.Value{
		Kind: &qdrant.Value_IntegerValue{
			IntegerValue: doc.UploadTime.Unix(),
		},
	}
	payload["description"] = &qdrant.Value{
		Kind: &qdrant.Value_StringValue{
			StringValue: doc.Description,
		},
	}

	points := []*qdrant.PointStruct{
		{
			Id: &qdrant.PointId{
				PointIdOptions: &qdrant.PointId_Uuid{
					Uuid: doc.ID,
				},
			},
			Vectors: &qdrant.Vectors{
				VectorsOptions: &qdrant.Vectors_Vector{
					Vector: &qdrant.Vector{
						Data: vector,
					},
				},
			},
			Payload: payload,
		},
	}

	_, err = pointsClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.config.DocumentMetadataCollection,
		Points:         points,
	})

	return err
}

func (s *QAService) ExtractPureText(response string) string {
	// 检查是否是 JSON 格式的响应
	if strings.HasPrefix(strings.TrimSpace(response), "{") {
		var result map[string]interface{}
		if err := json.Unmarshal([]byte(response), &result); err == nil {
			// 检查是否存在 "content" 字段
			if content, ok := result["content"].(string); ok {
				// 只返回第一个问题的答案
				answers := strings.Split(content, "Q：")
				if len(answers) > 1 {
					// 返回第一个答案，去掉 "A：" 前缀
					firstAnswer := strings.Split(answers[1], "A：")
					if len(firstAnswer) > 1 {
						return strings.TrimSpace(firstAnswer[1])
					}
				}
				return content
			}
		}
	}
	return response
}

// 添加一个新的方法来获取基于文档 ID 的检索器
func (s *QAService) getRetrieverForDocuments(documentIds []string) vectorstores.Retriever {
	if len(documentIds) == 0 {
		// 如果没有选择文档，使用所有文档
		return vectorstores.ToRetriever(s.qdrant.GetStore(), 10)
	}

	// 创建过滤器
	filter := &qdrant.Filter{
		Should: []*qdrant.Condition{
			{
				ConditionOneOf: &qdrant.Condition_HasId{
					HasId: &qdrant.HasIdCondition{
						HasId: &qdrant.HasIdCondition_OneOf{
							OneOf: &qdrant.RepeatedStrings{
								Strings: documentIds,
							},
						},
					},
				},
			},
		},
	}

	// 返回带过滤器的检索器
	return s.qdrant.GetStore().AsRetriever(10, vectorstores.WithFilters(filter))
}

// 修改 Query 方法
func (s *QAService) Query(ctx context.Context, question string, documentIds []string) (string, error) {
	retriever := s.getRetrieverForDocuments(documentIds)

	// 更新提示词
	prompt := fmt.Sprintf("请基于选中的文档，直接回答以下问题，不要列举其他相关问题。问题是：%s", question)

	qa := chains.NewRetrievalQAFromLLM(s.llm, retriever)

	response, err := chains.Run(ctx, qa, prompt)
	if err != nil {
		return "", err
	}

	return s.ExtractPureText(response), nil
}

// StreamResponse 结构体修改
type StreamResponse string

// StreamingQuery 方法修改
func (s *QAService) StreamingQuery(ctx context.Context, question string, documentIds []string, responseChan chan StreamResponse) error {
	// 获取基于选中文档的检索器
	retriever := s.getRetrieverForDocuments(documentIds)

	// 添加提示词
	prompt := fmt.Sprintf("请基于选中的文档，直接回答以下问题，不要列举其他相关问题。问题是：%s", question)

	qa := chains.NewRetrievalQAFromLLM(s.llm, retriever)

	go func() {
		defer close(responseChan)

		answer, err := chains.Run(ctx, qa, prompt) // 使用带提示词的问题
		if err != nil {
			select {
			case <-ctx.Done():
			case responseChan <- StreamResponse("Error: " + err.Error()):
			}
			return
		}

		pureText := s.ExtractPureText(answer)
		select {
		case <-ctx.Done():
			return
		case responseChan <- StreamResponse(pureText):
		}
	}()

	return nil
}

func (s *QAService) ListDocuments(ctx context.Context) (*models.DocumentList, error) {
	config := &qdrant.Config{
		Host: "localhost",
		Port: 6333,
	}

	client, err := qdrant.NewClient(config)
	if err != nil {
		return nil, fmt.Errorf("创建 Qdrant 客户端失败: %v", err)
	}

	var limit uint32 = 100
	request := &qdrant.ScrollPoints{
		CollectionName: s.config.DocumentMetadataCollection,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{
				Enable: true,
			},
		},
		Limit: &limit,
	}

	response, err := client.Scroll(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("获取文档列表失败: %v", err)
	}

	documents := make([]models.Document, 0)
	// 直接使用 response 作为点的切片
	for _, point := range response {
		doc := models.Document{
			ID:          point.Id.GetUuid(),
			Name:        point.Payload["name"].GetStringValue(),
			Type:        point.Payload["type"].GetStringValue(),
			Size:        point.Payload["size"].GetIntegerValue(),
			UploadTime:  time.Unix(point.Payload["upload_time"].GetIntegerValue(), 0),
			Description: point.Payload["description"].GetStringValue(),
		}
		documents = append(documents, doc)
	}

	return &models.DocumentList{
		Documents: documents,
		Total:     len(documents),
	}, nil
}
