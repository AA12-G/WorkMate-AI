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
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
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

	// 分割文档后，添加文档到向量存储
	docs := make([]schema.Document, len(chunkDocuments))
	docID := uuid.New().String() // 为整个文档生成一个唯一ID

	fmt.Printf("处理文件: %s, 文档ID: %s\n", filePath, docID)

	for i, doc := range chunkDocuments {
		// 为每个文档块添加唯一标识
		chunkID := fmt.Sprintf("%s_%d", docID, i)

		// 创建新的元数据映射
		metadata := map[string]interface{}{
			"doc_id":       docID,
			"chunk_id":     chunkID,
			"filename":     filepath.Base(filePath),
			"chunk_num":    i,
			"total_chunks": len(chunkDocuments),
		}

		// 打印调试信息
		fmt.Printf("处理文档块 %d/%d, 内容长度: %d, 元数据: %+v\n",
			i+1, len(chunkDocuments), len(doc.PageContent), metadata)

		docs[i] = schema.Document{
			PageContent: doc.PageContent,
			Metadata:    metadata,
		}
	}

	// 过滤掉空内容的文档
	validDocs := make([]schema.Document, 0)
	for _, doc := range docs {
		if len(doc.PageContent) > 0 {
			validDocs = append(validDocs, doc)
		}
	}

	// 添加文档到向量存储
	if len(validDocs) > 0 {
		_, err := s.qdrant.GetStore().AddDocuments(ctx, validDocs)
		if err != nil {
			return fmt.Errorf("添加文档到向量存储失败: %v", err)
		}
		fmt.Printf("成功添加 %d 个有效文档块\n", len(validDocs))
	} else {
		return fmt.Errorf("没有有效的文档内容可以添加")
	}

	return nil
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

// 修改检索器实现
func (s *QAService) getRetrieverForDocuments(documentIds []string) vectorstores.Retriever {
	if len(documentIds) == 0 {
		return vectorstores.ToRetriever(s.qdrant.GetStore(), 10)
	}

	// 暂时直接返回基础检索器，不做过滤
	return vectorstores.ToRetriever(s.qdrant.GetStore(), 10)
}

// 修改 Query 方法，添加错误处理
func (s *QAService) Query(ctx context.Context, question string, documentIds []string) (string, error) {
	fmt.Printf("用户选择的文档IDs: %v\n", documentIds)

	// 创建 gRPC 连接
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return "", fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)

	// 使用 Scroll API 获取所有文档
	var limit uint32 = 100
	var filtered []schema.Document
	var offset *qdrant.PointId = nil

	// 创建文档ID的映射，方便查找
	docMap := make(map[string]bool)
	for _, id := range documentIds {
		docMap[id] = true
	}

	for {
		request := &qdrant.ScrollPoints{
			CollectionName: s.config.CollectionName,
			Limit:          &limit,
			Offset:         offset,
			WithPayload: &qdrant.WithPayloadSelector{
				SelectorOptions: &qdrant.WithPayloadSelector_Enable{
					Enable: true,
				},
			},
		}

		response, err := client.Scroll(ctx, request)
		if err != nil {
			fmt.Printf("滚动获取文档时出错: %v\n", err)
			return "抱歉，检索文档时出现错误，请稍后再试。", nil
		}

		// 如果没有更多结果，退出循环
		if len(response.Result) == 0 {
			break
		}

		// 处理当前批次的文档
		for _, point := range response.Result {
			// 从 payload 中获取 doc_id
			if docIDValue, ok := point.Payload["doc_id"]; ok {
				docID := docIDValue.GetStringValue()
				if docMap[docID] {
					// 构建文档对象
					content := ""
					filename := ""

					if contentValue, ok := point.Payload["page_content"]; ok {
						content = contentValue.GetStringValue()
					}
					if filenameValue, ok := point.Payload["filename"]; ok {
						filename = filenameValue.GetStringValue()
					}

					if content != "" {
						doc := schema.Document{
							PageContent: content,
							Metadata: map[string]interface{}{
								"doc_id":   docID,
								"filename": filename,
							},
						}
						filtered = append(filtered, doc)
						fmt.Printf("找到匹配文档: ID=%s, 内容长度=%d\n", docID, len(content))
					}
				}
			}
		}

		// 更新 offset 用于下一次查询
		if len(response.Result) > 0 {
			offset = response.Result[len(response.Result)-1].Id
		}
	}

	// 打印调试信息
	fmt.Printf("\n检索结果统计:\n")
	fmt.Printf("- 找到 %d 个匹配文档\n", len(filtered))

	if len(filtered) == 0 {
		return "抱歉，在选择的文档中没有找到相关信息。", nil
	}

	// 构建上下文
	var context string
	for _, doc := range filtered {
		if name, ok := doc.Metadata["filename"].(string); ok {
			context += fmt.Sprintf("\n【来自文档：%s】\n", name)
		}
		context += doc.PageContent + "\n"
	}

	// 构建提示词
	prompt := fmt.Sprintf(`你是一个知识库问答助手。请仔细阅读以下内容，并基于这些内容回答问题。
如果内容中没有相关信息，请直接回复："抱歉，在当前内容中没有找到相关信息。"
不要编造或推测任何信息，只回答内容中明确提到的信息。

参考内容：
%s

问题：%s

请基于上述参考内容回答这个问题。如果内容中包含答案，请详细解答；如果没有相关信息，请明确指出。`, context, question)

	// 调用 LLM 生成回答
	llmResponse, err := s.llm.Call(ctx, prompt)
	if err != nil {
		fmt.Printf("生成回答时出错: %v\n", err)
		return "抱歉，AI 助手暂时无法回答您的问题，请稍后再试。", nil
	}

	return s.ExtractPureText(llmResponse), nil
}

// 添加辅助函数用于截断字符串
func truncateString(s string, length int) string {
	if len(s) <= length {
		return s
	}
	return s[:length] + "..."
}

// 添加新的辅助方法
func (s *QAService) getDocumentsFromMetadata(ctx context.Context, documentIds []string) ([]schema.Document, error) {
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)
	var documents []schema.Document

	for _, id := range documentIds {
		request := &qdrant.GetPoints{
			CollectionName: s.config.CollectionName,
			Ids: []*qdrant.PointId{
				{
					PointIdOptions: &qdrant.PointId_Uuid{
						Uuid: id,
					},
				},
			},
		}

		response, err := client.Get(ctx, request)
		if err != nil {
			fmt.Printf("获取文档 %s 失败: %v\n", id, err)
			continue
		}

		for _, point := range response.Result {
			if point.Payload != nil {
				content := point.Payload["content"].GetStringValue()
				if content != "" {
					doc := schema.Document{
						PageContent: content,
						Metadata: map[string]interface{}{
							"doc_id": id,
						},
					}
					documents = append(documents, doc)
				}
			}
		}
	}

	return documents, nil
}

// StreamResponse 结构体修改
type StreamResponse string

// StreamingQuery 方法修改
func (s *QAService) StreamingQuery(ctx context.Context, question string, documentIds []string, responseChan chan StreamResponse) error {
	// 如果没有选择文档，使用普通对话模式
	if len(documentIds) == 0 {
		go func() {
			defer close(responseChan)

			prompt := fmt.Sprintf(`你是一个智能助手。请回答用户的问题。
如果你不确定答案，请诚实地说"我不确定"或"我需要更多信息"。
不要编造或推测任何信息。

问题：%s`, question)

			response, err := s.llm.Call(ctx, prompt)
			if err != nil {
				fmt.Printf("生成回答时出错: %v\n", err)
				select {
				case <-ctx.Done():
				case responseChan <- StreamResponse("抱歉，AI 助手暂时无法回答您的问题，请稍后再试。"):
				}
				return
			}

			select {
			case <-ctx.Done():
			case responseChan <- StreamResponse(s.ExtractPureText(response)):
			}
		}()
		return nil
	}

	// 以下是基于知识库的问答逻辑
	fmt.Printf("用户选择的文档IDs: %v\n", documentIds)

	go func() {
		defer close(responseChan)

		// 使用 SimilaritySearch 而不是 GetRelevantDocuments
		allDocs, err := s.qdrant.GetStore().SimilaritySearch(ctx, question, 100)
		if err != nil {
			fmt.Printf("检索文档时出错: %v\n", err)
			select {
			case <-ctx.Done():
			case responseChan <- StreamResponse("抱歉，检索文档时出现错误，请稍后再试。"):
			}
			return
		}

		// 过滤文档，只保留选中的文档
		filtered := make([]schema.Document, 0)
		docMap := make(map[string]bool)
		for _, id := range documentIds {
			docMap[id] = true
		}

		for _, doc := range allDocs {
			docID, ok := doc.Metadata["doc_id"].(string)
			if !ok {
				continue
			}
			if docMap[docID] {
				filtered = append(filtered, doc)
				fmt.Printf("匹配到文档: %s, 内容长度: %d\n", docID, len(doc.PageContent))
			}
		}

		// 打印调试信息
		fmt.Printf("检索到 %d 个文档，过滤后剩余 %d 个文档\n", len(allDocs), len(filtered))

		if len(filtered) == 0 {
			select {
			case <-ctx.Done():
			case responseChan <- StreamResponse("抱歉，在选择的文档中没有找到相关信息。"):
			}
			return
		}

		// 构建上下文
		var context string
		for _, doc := range filtered {
			if name, ok := doc.Metadata["filename"].(string); ok {
				context += fmt.Sprintf("\n【来自文档：%s】\n", name)
			}
			context += doc.PageContent + "\n"
		}

		// 构建提示词
		prompt := fmt.Sprintf(`你是一个知识库问答助手。请仔细阅读以下内容，并基于这些内容回答问题。
如果内容中没有相关信息，请直接回复："抱歉，在当前内容中没有找到相关信息。"
不要编造或推测任何信息，只回答内容中明确提到的信息。

参考内容：
%s

问题：%s

请基于上述参考内容回答这个问题。如果内容中包含答案，请详细解答；如果没有相关信息，请明确指出。`, context, question)

		// 生成回答
		response, err := s.llm.Call(ctx, prompt)
		if err != nil {
			fmt.Printf("生成回答时出错: %v\n", err)
			select {
			case <-ctx.Done():
			case responseChan <- StreamResponse("抱歉，AI 助手暂时无法回答您的问题，请稍后再试。"):
			}
			return
		}

		select {
		case <-ctx.Done():
		case responseChan <- StreamResponse(s.ExtractPureText(response)):
			fmt.Printf("已发送回答\n")
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
