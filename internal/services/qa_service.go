package services

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"WorkMate-AI/config"
	"WorkMate-AI/internal/models"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/textsplitter"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type QAService struct {
	llm    llms.LLM
	config *config.Config
}

// 文档块结构
type DocumentChunk struct {
	ID       string
	DocID    string
	Content  string
	Filename string
	ChunkNum int
}

// StreamResponse 定义流式响应的结构
type StreamResponse struct {
	Content string `json:"content"`
}

func NewQAService(cfg *config.Config) (*QAService, error) {
	llmInstance, err := ollama.New(
		ollama.WithModel(cfg.OllamaModel),
		ollama.WithSystemPrompt(`你是一个严格的知识库问答助手。你必须遵守以下规则：
1. 只能使用提供的文档内容回答问题
2. 回答必须以"根据文档《xxx》"开头
3. 不允许编造或推测任何未在文档中明确提到的信息
4. 如果文档中没有相关信息，必须明确说明
5. 保持回答的准确性和客观性
6. 直接引用文档中的原文，不要随意改写
7. 如果信息来自多个文档，需要分别标明来源`),
	)
	if err != nil {
		return nil, err
	}

	return &QAService{
		llm:    llmInstance,
		config: cfg,
	}, nil
}

// ProcessFile 处理文件并存储到 Qdrant
func (s *QAService) ProcessFile(ctx context.Context, filePath string) error {
	fmt.Printf("开始处理文件: %s\n", filePath)

	// 使用更大的批次大小来减少网络请求
	batchSize := 20
	waitTrue := true

	// 连接到 Qdrant
	conn, err := grpc.Dial(s.config.QdrantGRPCURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(50*1024*1024)),
	)
	if err != nil {
		return fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	// 检查并创建集合
	collectionsClient := qdrant.NewCollectionsClient(conn)
	_, err = collectionsClient.Get(ctx, &qdrant.GetCollectionInfoRequest{
		CollectionName: s.config.CollectionName,
	})
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			// 创建集合
			createRequest := &qdrant.CreateCollection{
				CollectionName: s.config.CollectionName,
				VectorsConfig: &qdrant.VectorsConfig{
					Config: &qdrant.VectorsConfig_Params{
						Params: &qdrant.VectorParams{
							Size:     3072,
							Distance: qdrant.Distance_Cosine,
						},
					},
				},
			}
			_, err = collectionsClient.Create(ctx, createRequest)
			if err != nil {
				return fmt.Errorf("创建集合失败: %v", err)
			}
			fmt.Printf("成功创建集合: %s\n", s.config.CollectionName)
		} else {
			return fmt.Errorf("检查集合失败: %v", err)
		}
	}

	// 使用 goroutine 并发处理文本分块
	type chunkResult struct {
		chunks []string
		err    error
	}
	chunkChan := make(chan chunkResult)

	go func() {
		// 读取和处理文件
		content, err := os.ReadFile(filePath)
		if err != nil {
			chunkChan <- chunkResult{err: fmt.Errorf("读取文件失败: %v", err)}
			return
		}

		// 清理和分割文本
		textContent := cleanText(string(content))
		if !isValidUTF8(textContent) {
			chunkChan <- chunkResult{err: fmt.Errorf("文件内容包含无效的字符编码")}
			return
		}

		splitter := textsplitter.NewRecursiveCharacter(
			textsplitter.WithChunkSize(2000),
			textsplitter.WithChunkOverlap(200),
		)

		chunks, err := splitter.SplitText(textContent)
		if err != nil {
			chunkChan <- chunkResult{err: fmt.Errorf("分割文本失败: %v", err)}
			return
		}

		// 过滤空白块
		validChunks := make([]string, 0, len(chunks))
		for _, chunk := range chunks {
			if cleaned := strings.TrimSpace(chunk); cleaned != "" {
				validChunks = append(validChunks, cleaned)
			}
		}

		chunkChan <- chunkResult{chunks: validChunks}
	}()

	// 同时连接到 Qdrant
	conn, err = grpc.Dial(s.config.QdrantGRPCURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(50*1024*1024)), // 增加消息大小限制
	)
	if err != nil {
		return fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	// 等待文本处理结果
	result := <-chunkChan
	if result.err != nil {
		return result.err
	}

	validChunks := result.chunks
	if len(validChunks) == 0 {
		return fmt.Errorf("没有提取到有效内容")
	}

	fmt.Printf("成功提取 %d 个有效文档块\n", len(validChunks))

	// 生成文档ID
	docID := uuid.New().String()
	filename := filepath.Base(filePath)

	// 预先生成所有向量
	vectors := make([][]float32, len(validChunks))
	for i := range vectors {
		vectors[i] = make([]float32, 3072)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}

	// 使用工作池并发上传
	workerCount := 3
	workChan := make(chan int, len(validChunks))
	errChan := make(chan error, workerCount)
	var wg sync.WaitGroup

	// 启动工作协程
	for w := 0; w < workerCount; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pointsClient := qdrant.NewPointsClient(conn)

			for i := range workChan {
				end := i + batchSize
				if end > len(validChunks) {
					end = i + (len(validChunks) - i)
				}

				batchPoints := make([]*qdrant.PointStruct, 0, end-i)
				for j := i; j < end; j++ {
					// 为每个块生成唯一的 UUID
					chunkID := uuid.New().String()
					point := &qdrant.PointStruct{
						Id: &qdrant.PointId{
							PointIdOptions: &qdrant.PointId_Uuid{
								Uuid: chunkID,
							},
						},
						Vectors: &qdrant.Vectors{
							VectorsOptions: &qdrant.Vectors_Vector{
								Vector: &qdrant.Vector{
									Data: vectors[j],
								},
							},
						},
						Payload: map[string]*qdrant.Value{
							"doc_id":    {Kind: &qdrant.Value_StringValue{StringValue: docID}},
							"content":   {Kind: &qdrant.Value_StringValue{StringValue: validChunks[j]}},
							"filename":  {Kind: &qdrant.Value_StringValue{StringValue: filename}},
							"chunk_num": {Kind: &qdrant.Value_IntegerValue{IntegerValue: int64(j)}},
							"chunk_id":  {Kind: &qdrant.Value_StringValue{StringValue: chunkID}}, // 添加 chunk_id 到 payload
						},
					}
					batchPoints = append(batchPoints, point)
				}

				// 重试逻辑
				var uploadErr error
				for retry := 0; retry < 3; retry++ {
					_, uploadErr = pointsClient.Upsert(ctx, &qdrant.UpsertPoints{
						CollectionName: s.config.CollectionName,
						Points:         batchPoints,
						Wait:           &waitTrue,
					})
					if uploadErr == nil {
						fmt.Printf("成功上传批次 %d-%d\n", i, end-1)
						break
					}
					if retry < 2 {
						time.Sleep(time.Second * time.Duration(retry+1))
					}
				}
				if uploadErr != nil {
					errChan <- fmt.Errorf("上传批次 %d-%d 失败: %v", i, end-1, uploadErr)
					return
				}
			}
		}()
	}

	// 分发工作
	for i := 0; i < len(validChunks); i += batchSize {
		workChan <- i
	}
	close(workChan)

	// 等待所有工作完成
	wg.Wait()
	close(errChan)

	// 检查错误
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	fmt.Printf("成功处理文件 %s，文档ID: %s，共 %d 个块\n", filename, docID, len(validChunks))
	return nil
}

// cleanText 清理文本内容
func cleanText(text string) string {
	// 1. 移除不可打印字符，但保留基本标点和空白
	var result strings.Builder
	for _, r := range text {
		if unicode.IsPrint(r) || unicode.IsSpace(r) || unicode.IsPunct(r) {
			result.WriteRune(r)
		}
	}
	text = result.String()

	// 2. 规范化空白字符，但保留段落
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = strings.Join(strings.Fields(line), " ")
	}
	text = strings.Join(lines, "\n")

	// 3. 移除连续的空行
	text = strings.ReplaceAll(text, "\n\n\n", "\n\n")

	// 4. 移除 UTF-8 BOM
	text = strings.TrimPrefix(text, "\uFEFF")

	return text
}

// 验证 UTF-8
func isValidUTF8(s string) bool {
	return utf8.ValidString(s)
}

// Query 查询文档
func (s *QAService) Query(ctx context.Context, question string, documentIds []string) (string, error) {
	// 检查是否是普通对话或没有选择文档
	if isGeneralConversation(question) || len(documentIds) == 0 {
		return handleGeneralConversation(question), nil
	}

	fmt.Printf("开始查询，问题：%s，文档IDs：%v\n", question, documentIds)

	// 连接到 Qdrant
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return "", fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)

	// 构建查询条件
	var conditions []*qdrant.Condition
	for _, docID := range documentIds {
		conditions = append(conditions, &qdrant.Condition{
			ConditionOneOf: &qdrant.Condition_Field{
				Field: &qdrant.FieldCondition{
					Key: "doc_id",
					Match: &qdrant.Match{
						MatchValue: &qdrant.Match_Keyword{
							Keyword: docID,
						},
					},
				},
			},
		})
	}

	// 构建查询
	searchRequest := &qdrant.SearchPoints{
		CollectionName: s.config.CollectionName,
		Vector:         make([]float32, 3072), // 使用零向量
		Filter: &qdrant.Filter{
			Should: conditions, // 使用 should 来匹配任意一个文档ID
		},
		Limit: 100,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{
				Enable: true,
			},
		},
	}

	fmt.Printf("执行搜索，集合：%s\n", s.config.CollectionName)

	// 执行搜索
	response, err := client.Search(ctx, searchRequest)
	if err != nil {
		return "", fmt.Errorf("搜索文档失败: %v", err)
	}

	fmt.Printf("搜索结果数量：%d\n", len(response.Result))

	// 构建上下文
	var context strings.Builder
	var docContents = make(map[string][]string) // 用于按文档组织内容

	// 首先按文档ID组织内容
	for _, point := range response.Result {
		docID := point.Payload["doc_id"].GetStringValue()
		content := point.Payload["content"].GetStringValue()
		if content != "" {
			docContents[docID] = append(docContents[docID], content)
		}
	}

	// 按文档组织内容
	for docID, contents := range docContents {
		filename := ""
		for _, point := range response.Result {
			if point.Payload["doc_id"].GetStringValue() == docID {
				filename = point.Payload["filename"].GetStringValue()
				break
			}
		}
		context.WriteString(fmt.Sprintf("\n文档《%s》内容：\n", filename))
		for _, content := range contents {
			context.WriteString(content)
			context.WriteString("\n")
		}
		context.WriteString("\n---\n")
	}

	if context.Len() == 0 {
		return "未找到相关文档内容。", nil
	}

	// 构建提示词
	prompt := fmt.Sprintf(`你是一个专业的知识库问答助手。请严格按照以下要求回答问题：

===== 文档内容开始 =====
%s
===== 文档内容结束 =====

用户问题：%s

严格要求：
1. 必须以"根据文档《xxx》"开头回答
2. 只能使用文档中明确提到的信息回答
3. 如果文档中没有相关信息，必须回答"抱歉，在所选文档中没有找到相关信息"
4. 不允许编造、推测或补充任何文档中未提到的信息
5. 如果要引用多个文档，需要分别指明来源
6. 回答要简洁准确，直接引用相关内容

请严格按照上述要求回答问题：`,
		context.String(), question)

	fmt.Printf("文档内容：\n%s\n", context.String())
	fmt.Printf("提示词：\n%s\n", prompt)

	fmt.Println("开始生成回答...")

	// 生成回答
	answer, err := s.llm.Call(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("生成回答失败: %v", err)
	}

	fmt.Println("回答生成完成")
	return answer, nil
}

// 判断是否是普通对话
func isGeneralConversation(question string) bool {
	// 定义一些普通对话的关键词
	generalPhrases := []string{
		"你好", "hi", "hello", "嗨",
		"介绍", "自我介绍",
		"再见", "拜拜", "goodbye",
		"早上好", "晚上好", "下午好",
		"谢谢", "感谢",
	}

	questionLower := strings.ToLower(question)
	for _, phrase := range generalPhrases {
		if strings.Contains(questionLower, strings.ToLower(phrase)) {
			return true
		}
	}
	return false
}

// 处理普通对话
func handleGeneralConversation(question string) string {
	questionLower := strings.ToLower(question)

	switch {
	case strings.Contains(questionLower, "你好") ||
		strings.Contains(questionLower, "hi") ||
		strings.Contains(questionLower, "hello"):
		return "你好！我是 AI 智能办公助手。我可以帮你查询和理解文档内容，请告诉我你想了解什么？"

	case strings.Contains(questionLower, "介绍"):
		return "我是一个 AI 智能办公助手，专门用于帮助用户理解和分析文档内容。你可以上传文档，然后向我提问，我会基于文档内容为你解答。需要我为你做些什么吗？"

	case strings.Contains(questionLower, "再见") ||
		strings.Contains(questionLower, "拜拜"):
		return "再见！如果还有问题随时问我。"

	case strings.Contains(questionLower, "谢谢") ||
		strings.Contains(questionLower, "感谢"):
		return "不用谢！很高兴能帮到你。还有其他问题吗？"

	default:
		return "你好！我是 AI 助手。我主要负责解答与文档相关的问题。你可以上传文档，然后问我任何关于文档内容的问题。请问有什么我可以帮你的吗？"
	}
}

// StreamingQuery 处理流式查询
func (s *QAService) StreamingQuery(ctx context.Context, question string, documentIds []string, responseChan chan<- StreamResponse) error {
	// 检查是否是普通对话或没有选择文档
	if isGeneralConversation(question) || len(documentIds) == 0 {
		select {
		case <-ctx.Done():
		case responseChan <- StreamResponse{handleGeneralConversation(question)}:
		}
		return nil
	}

	fmt.Printf("开始流式查询，问题：%s，文档IDs：%v\n", question, documentIds)

	// 连接到 Qdrant
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)

	// 构建查询条件
	var conditions []*qdrant.Condition
	for _, docID := range documentIds {
		conditions = append(conditions, &qdrant.Condition{
			ConditionOneOf: &qdrant.Condition_Field{
				Field: &qdrant.FieldCondition{
					Key: "doc_id",
					Match: &qdrant.Match{
						MatchValue: &qdrant.Match_Keyword{
							Keyword: docID,
						},
					},
				},
			},
		})
	}

	// 构建查询
	searchRequest := &qdrant.SearchPoints{
		CollectionName: s.config.CollectionName,
		Vector:         make([]float32, 3072),
		Filter: &qdrant.Filter{
			Should: conditions,
		},
		Limit: 100,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{
				Enable: true,
			},
		},
	}

	fmt.Printf("执行流式搜索，集合：%s\n", s.config.CollectionName)

	// 执行搜索
	response, err := client.Search(ctx, searchRequest)
	if err != nil {
		return fmt.Errorf("搜索文档失败: %v", err)
	}

	fmt.Printf("流式搜索结果数量：%d\n", len(response.Result))

	// 构建上下文
	var context strings.Builder
	var docContents = make(map[string][]string)

	// 首先按文档ID组织内容
	for _, point := range response.Result {
		docID := point.Payload["doc_id"].GetStringValue()
		content := point.Payload["content"].GetStringValue()
		if content != "" {
			docContents[docID] = append(docContents[docID], content)
		}
	}

	// 按文档组织内容
	for docID, contents := range docContents {
		filename := ""
		for _, point := range response.Result {
			if point.Payload["doc_id"].GetStringValue() == docID {
				filename = point.Payload["filename"].GetStringValue()
				break
			}
		}
		context.WriteString(fmt.Sprintf("\n文档《%s》内容：\n", filename))
		for _, content := range contents {
			context.WriteString(content)
			context.WriteString("\n")
		}
		context.WriteString("\n---\n")
	}

	if context.Len() == 0 {
		select {
		case <-ctx.Done():
		case responseChan <- StreamResponse{"未找到相关文档内容。"}:
		}
		return nil
	}

	// 构建提示词
	prompt := fmt.Sprintf(`请基于以下文档内容回答用户问题。

===== 文档内容开始 =====
%s
===== 文档内容结束 =====

用户问题：%s

注意事项：
1. 必须以"根据文档《xxx》，..."的格式开始回答
2. 只能使用上述文档中的信息
3. 如果文档中没有相关信息，请回答"抱歉，在所选文档中没有找到相关信息"
4. 回答要简洁明了，直接引用相关内容
5. 不要编造或推测任何文档中没有的信息

请开始回答：`,
		context.String(), question)

	fmt.Println("开始生成流式回答...")

	// 生成回答
	answer, err := s.llm.Call(ctx, prompt)
	if err != nil {
		return fmt.Errorf("生成回答失败: %v", err)
	}

	fmt.Println("流式回答生成完成")

	select {
	case <-ctx.Done():
	case responseChan <- StreamResponse{answer}:
	}

	return nil
}

// ListDocuments 获取文档列表
func (s *QAService) ListDocuments(ctx context.Context) (*models.DocumentList, error) {
	// 连接到 Qdrant
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)

	// 使用 Scroll API 获取所有文档
	var limit uint32 = 100
	request := &qdrant.ScrollPoints{
		CollectionName: s.config.CollectionName,
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

	// 使用 map 来去重文档
	docMap := make(map[string]*models.Document)
	for _, point := range response.Result {
		if docID := point.Payload["doc_id"].GetStringValue(); docID != "" {
			if _, exists := docMap[docID]; !exists {
				// 从 payload 中获取上传时间，如果没有则使用当前时间
				var uploadTime time.Time
				if timeStr := point.Payload["upload_time"].GetStringValue(); timeStr != "" {
					uploadTime, _ = time.Parse(time.RFC3339, timeStr)
				} else {
					uploadTime = time.Now()
				}

				docMap[docID] = &models.Document{
					ID:         docID,
					Name:       point.Payload["filename"].GetStringValue(),
					UploadTime: uploadTime,
				}
			}
		}
	}

	// 转换为切片
	docs := make([]models.Document, 0, len(docMap))
	for _, doc := range docMap {
		docs = append(docs, *doc)
	}

	return &models.DocumentList{
		Documents: docs,
		Total:     len(docs),
	}, nil
}

// DeleteDocument 删除文档
func (s *QAService) DeleteDocument(ctx context.Context, docID string) error {
	// 连接到 Qdrant
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)

	// 构建删除条件
	var waitTrue bool = true
	deleteRequest := &qdrant.DeletePoints{
		CollectionName: s.config.CollectionName,
		Wait:           &waitTrue,
		Points: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Filter{
				Filter: &qdrant.Filter{
					Must: []*qdrant.Condition{
						{
							ConditionOneOf: &qdrant.Condition_Field{
								Field: &qdrant.FieldCondition{
									Key: "doc_id",
									Match: &qdrant.Match{
										MatchValue: &qdrant.Match_Keyword{
											Keyword: docID,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	// 执行删除
	_, err = client.Delete(ctx, deleteRequest)
	if err != nil {
		return fmt.Errorf("删除文档失败: %v", err)
	}

	return nil
}
