package services

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"WorkMate-AI/config"
	"WorkMate-AI/internal/models"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/textsplitter"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/transform"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// 正则表达式常量
const (
	controlCharsPattern = `[\x00-\x1F\x7F]`
	multiSpacePattern   = `\s+`
	basicCleanPattern   = `[^\p{Han}\p{Latin}\p{N}\s,.?!，。？！、：；""''（）()《》\[\]\-]`
)

// 预编译的正则表达式
var (
	controlCharsRegex = regexp.MustCompile(controlCharsPattern)
	multiSpaceRegex   = regexp.MustCompile(multiSpacePattern)
	basicCleanRegex   = regexp.MustCompile(basicCleanPattern)
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

	// 读取文件
	content, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("读取文件失败: %v", err)
	}

	// 添加日志查看文件内容
	fmt.Printf("原始文件内容长度: %d\n", len(content))
	if len(content) > 100 {
		fmt.Printf("文件前100个字符: %s\n", content[:100])
	}

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
		// 改进编码检测逻辑
		textContent, err := detectAndConvertEncoding(content)
		if err != nil {
			chunkChan <- chunkResult{err: err}
			return
		}

		// 清理文本
		textContent = cleanText(textContent)

		// 验证清理后的文本
		if !utf8.ValidString(textContent) {
			chunkChan <- chunkResult{err: fmt.Errorf("清理后的文本包含无效字符")}
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

		// 添加日志
		fmt.Printf("分块数量: %d\n", len(chunks))
		if len(chunks) > 0 {
			fmt.Printf("第一个分块内容: %s\n", chunks[0])
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

	// 获取文件的实际修改时间
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("获取文件信息失败: %v", err)
	}

	// 使用文件的修改时间作为上传时间
	uploadTime := fileInfo.ModTime().Format(time.RFC3339)

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
							"doc_id":      {Kind: &qdrant.Value_StringValue{StringValue: docID}},
							"content":     {Kind: &qdrant.Value_StringValue{StringValue: validChunks[j]}},
							"filename":    {Kind: &qdrant.Value_StringValue{StringValue: filename}},
							"chunk_num":   {Kind: &qdrant.Value_IntegerValue{IntegerValue: int64(j)}},
							"chunk_id":    {Kind: &qdrant.Value_StringValue{StringValue: chunkID}},
							"upload_time": {Kind: &qdrant.Value_StringValue{StringValue: uploadTime}},
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
						fmt.Printf("成功上传批次 %d-%d，每个点的内容：\n", i, end-1)
						for _, point := range batchPoints {
							fmt.Printf("- 文档ID: %s\n", point.Payload["doc_id"].GetStringValue())
							fmt.Printf("- 文件名: %s\n", point.Payload["filename"].GetStringValue())
							contentPreview := point.Payload["content"].GetStringValue()
							if len(contentPreview) > 50 {
								contentPreview = contentPreview[:50] + "..."
							}
							fmt.Printf("- 内容预览: %s\n", contentPreview)
						}
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

// 改进编码检测逻辑
func detectAndConvertEncoding(content []byte) (string, error) {
	// 首先尝试 UTF-8
	if utf8.Valid(content) {
		return string(content), nil
	}

	// 尝试 GBK
	reader := transform.NewReader(bytes.NewReader(content), simplifiedchinese.GBK.NewDecoder())
	if decoded, err := io.ReadAll(reader); err == nil {
		if utf8.Valid(decoded) {
			return string(decoded), nil
		}
	}

	// 尝试 GB18030
	reader = transform.NewReader(bytes.NewReader(content), simplifiedchinese.GB18030.NewDecoder())
	if decoded, err := io.ReadAll(reader); err == nil {
		if utf8.Valid(decoded) {
			return string(decoded), nil
		}
	}

	return "", fmt.Errorf("无法识别文件编码")
}

// 修改 cleanText 函数，改进文本处理逻辑
func cleanText(text string) string {
	// 移除 BOM
	text = strings.TrimPrefix(text, "\uFEFF")

	// 移除所有特殊字符和数字标记
	text = regexp.MustCompile(`[*\[\]【】\-:：\d\.]+`).ReplaceAllString(text, "")

	// 移除所有非中文、英文、基本标点的字符
	text = regexp.MustCompile(`[^\p{Han}\p{Latin}，。？！：；、（）《》\s]+`).ReplaceAllString(text, "")

	// 处理英文单词，确保它们之间有空格
	text = regexp.MustCompile(`([a-zA-Z])([^\sa-zA-Z])`).ReplaceAllString(text, "$1 $2")
	text = regexp.MustCompile(`([^\sa-zA-Z])([a-zA-Z])`).ReplaceAllString(text, "$1 $2")

	// 处理重复的标点符号
	text = regexp.MustCompile(`[，。？！：；、]{2,}`).ReplaceAllString(text, "。")

	// 分段处理并去重
	paragraphs := strings.Split(text, "\n")
	var cleanedParagraphs []string
	seenContent := make(map[string]bool)

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		// 分句处理
		sentences := strings.Split(para, "。")
		var uniqueSentences []string

		for _, sentence := range sentences {
			sentence = strings.TrimSpace(sentence)
			// 只保留有意义的句子（至少包含一个中文字符）
			if sentence != "" && regexp.MustCompile(`\p{Han}`).MatchString(sentence) {
				// 去重
				if !seenContent[sentence] {
					seenContent[sentence] = true
					uniqueSentences = append(uniqueSentences, sentence)
				}
			}
		}

		if len(uniqueSentences) > 0 {
			cleanedParagraphs = append(cleanedParagraphs, strings.Join(uniqueSentences, "。"))
		}
	}

	return strings.Join(cleanedParagraphs, "\n")
}

// 验证 UTF-8
func isValidUTF8(s string) bool {
	return utf8.ValidString(s)
}

// Query 查询文档
func (s *QAService) Query(ctx context.Context, question string, documentIds []string) (string, error) {
	fmt.Printf("开始查询，问题：%s\n", question)
	fmt.Printf("选择的文档IDs：%v\n", documentIds)

	// 检查是否是普通对话
	if isGeneralConversation(question) {
		return handleGeneralConversation(question), nil
	}

	// 检查文档ID是否为空
	if len(documentIds) == 0 {
		return `请先选择要查询的文档。

您可以：
1. 点击左侧文档列表中的文档进行选择
2. 如果没有看到文档，请先上传文档
3. 选择文档后再次提问

提示：您可以通过点击文档名称前的复选框来选择文档。`, nil
	}

	// 检查文档ID是否有效
	if documentIds[0] == "" {
		return `您选择的文档似乎无效。

请确保：
1. 已经正确选择了文档（通过点击复选框）
2. 文档已经成功上传
3. 文档内容不为空

如果问题仍然存在，请尝试重新上传文档。`, nil
	}

	// 连接到 Qdrant
	conn, err := grpc.Dial(s.config.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return "", fmt.Errorf("连接 Qdrant 失败: %v", err)
	}
	defer conn.Close()

	client := qdrant.NewPointsClient(conn)

	// 修改搜索请求
	searchRequest := &qdrant.SearchPoints{
		CollectionName: s.config.CollectionName,
		Vector:         make([]float32, 3072),
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				{
					ConditionOneOf: &qdrant.Condition_Field{
						Field: &qdrant.FieldCondition{
							Key: "doc_id",
							Match: &qdrant.Match{
								MatchValue: &qdrant.Match_Keyword{
									Keyword: documentIds[0], // 先只查询第一个文档
								},
							},
						},
					},
				},
			},
		},
		Limit: 100,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{
				Enable: true,
			},
		},
	}

	// 执行搜索并打印详细日志
	fmt.Printf("执行搜索，请求：%+v\n", searchRequest)
	response, err := client.Search(ctx, searchRequest)
	if err != nil {
		return "", fmt.Errorf("搜索文档失败: %v", err)
	}

	fmt.Printf("搜索结果数量：%d\n", len(response.Result))

	// 如果没有找到结果，尝试使用 Scroll API
	if len(response.Result) == 0 {
		fmt.Println("使用 Scroll API 重试...")
		var limit uint32 = 100
		scrollRequest := &qdrant.ScrollPoints{
			CollectionName: s.config.CollectionName,
			Filter: &qdrant.Filter{
				Must: []*qdrant.Condition{
					{
						ConditionOneOf: &qdrant.Condition_Field{
							Field: &qdrant.FieldCondition{
								Key: "doc_id",
								Match: &qdrant.Match{
									MatchValue: &qdrant.Match_Keyword{
										Keyword: documentIds[0],
									},
								},
							},
						},
					},
				},
			},
			Limit: &limit,
			WithPayload: &qdrant.WithPayloadSelector{
				SelectorOptions: &qdrant.WithPayloadSelector_Enable{
					Enable: true,
				},
			},
		}

		scrollResponse, err := client.Scroll(ctx, scrollRequest)
		if err != nil {
			return "", fmt.Errorf("滚动搜索失败: %v", err)
		}

		// 将 RetrievedPoint 转换为 ScoredPoint
		convertedResults := make([]*qdrant.ScoredPoint, len(scrollResponse.Result))
		for i, point := range scrollResponse.Result {
			convertedResults[i] = &qdrant.ScoredPoint{
				Id:      point.Id,
				Payload: point.Payload,
				Score:   1.0, // 设置一个默认分数
			}
		}
		response.Result = convertedResults
		fmt.Printf("Scroll API 结果数量：%d\n", len(response.Result))
	}

	// 构建上下文
	var context strings.Builder
	var docContents = make(map[string][]string) // 用于按文档组织内容

	// 首先按文档ID组织内容
	for _, point := range response.Result {
		docID := point.Payload["doc_id"].GetStringValue()
		content := point.Payload["content"].GetStringValue()
		filename := point.Payload["filename"].GetStringValue()
		fmt.Printf("处理文档块：文档ID=%s, 文件名=%s\n", docID, filename)
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
			cleanedContent := cleanText(content)

			// 对于简历类文档，使用更严格的结构化处理
			if strings.Contains(filename, "简历") || strings.Contains(question, "简历") {
				sections := map[string][]string{
					"基本信息": {"姓名", "毕业", "学校", "专业"},
					"教育背景": {"学习", "专业", "课程"},
					"技术技能": {"技术", "开发", "框架", "语言", "数据库"},
					"工作经验": {"工作", "负责", "项目", "系统"},
					"个人特点": {"性格", "能力", "特点", "擅长"},
				}

				var structuredContent strings.Builder
				for section, keywords := range sections {
					var sectionContent []string
					sentences := strings.Split(cleanedContent, "。")

					for _, sentence := range sentences {
						sentence = strings.TrimSpace(sentence)
						for _, keyword := range keywords {
							if strings.Contains(sentence, keyword) {
								sectionContent = append(sectionContent, sentence)
								break
							}
						}
					}

					if len(sectionContent) > 0 {
						structuredContent.WriteString(section + "：\n")
						for _, s := range sectionContent {
							structuredContent.WriteString("- " + s + "\n")
						}
						structuredContent.WriteString("\n")
					}
				}

				cleanedContent = structuredContent.String()
			}

			if cleanedContent != "" {
				context.WriteString(cleanedContent)
				context.WriteString("\n")
			}
		}
		context.WriteString("\n---\n")
	}

	if context.Len() == 0 {
		return "未找到相关文档内容。", nil
	}

	// 修改提示词
	prompt := fmt.Sprintf(`你是一个专业的知识库问答助手。请基于以下文档内容回答问题：

===== 文档内容开始 =====
%s
===== 文档内容结束 =====

用户问题：%s

严格要求：
1. 必须以"根据文档《xxx》"开头
2. 回答必须结构化，使用以下格式：
   - 对于简历类文档：按照"教育背景"、"专业技能"、"工作经验"、"个人特点"等分类
   - 对于技术文档：按照"功能特点"、"技术架构"、"应用场景"等分类
   - 对于业务文档：按照"业务流程"、"关键概念"、"注意事项"等分类
3. 每个分类下的内容要：
   - 使用序号或要点符号列举
   - 保持逻辑顺序
   - 语言清晰简洁
4. 如果文档内容有重复或混乱：
   - 去除重复内容
   - 按逻辑重新组织
   - 保持内容的完整性
5. 确保回答的可读性：
   - 适当使用段落分隔
   - 保持格式统一
   - 突出重点内容

请开始回答：`, context.String(), question)

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

// 修改 isGeneralConversation 函数，增加更多的普通对话识别
func isGeneralConversation(question string) bool {
	generalPhrases := []string{
		"你好", "hi", "hello", "嗨",
		"再见", "拜拜", "goodbye",
		"在吗", "在么", "在不在",
	}

	questionLower := strings.TrimSpace(strings.ToLower(question))
	for _, phrase := range generalPhrases {
		if questionLower == strings.ToLower(phrase) {
			return true
		}
	}
	return false
}

// 处理普通对话
func handleGeneralConversation(question string) string {
	// 使用更专业的回复模板
	const greeting = `你好！我是一个专业的知识库问答助手。

我可以为您提供以下服务：
1. 回答通用问题和基础咨询
2. 基于文档的专业知识问答：
   - 技术文档解析和查询
   - 业务文档内容分析
   - 多文档交叉引用和对比
3. 文档内容的精确检索和分析

请选择相关文档或直接提出您的问题，我会为您提供专业的解答。`

	const springInfo = `Spring 是一个功能强大的开源应用开发框架，主要用于构建企业级 Java 应用。它具有以下核心特点：

1. 依赖注入（DI）和控制反转（IoC）
2. 面向切面编程（AOP）支持
3. 声明式事务管理
4. 强大的 MVC 架构支持
5. 灵活的数据访问抽象层

如果您想了解更多具体细节，建议选择相关的 Spring 技术文档，我可以为您提供深入的技术解答。`

	const defaultReply = `我是一个专业的知识库问答助手，可以为您提供以下服务：

1. 文档内容精确检索
2. 多维度文档分析
3. 专业知识问答
4. 技术文档解读
5. 业务文档分析

请选择您感兴趣的文档，我会基于文档内容为您提供专业、准确的解答。`

	// 使用简单的条件判断返回固定回复
	if strings.Contains(strings.ToLower(question), "你好") {
		return greeting
	}

	if strings.Contains(strings.ToLower(question), "spring") {
		return springInfo
	}

	return defaultReply
}

// StreamingQuery 处理流式查询
func (s *QAService) StreamingQuery(ctx context.Context, question string, documentIds []string, responseChan chan<- StreamResponse) error {
	// 检查是否是普通对话或没有选择文档
	if isGeneralConversation(question) || len(documentIds) == 0 {
		response := `您好！我是一个专业的知识库问答助手，很高兴为您服务。

我可以为您提供以下专业服务：
1. 智能文档解析与分析
   - 技术文档精准解读
   - 业务文档深度分析
   - 多文档交叉对比

2. 专业知识问答服务
   - 基于文档的精确答疑
   - 多维度内容关联
   - 上下文智能理解

3. 文档内容检索
   - 精准定位关键信息
   - 智能匹配相关内容
   - 多层次信息提取

请选择您感兴趣的文档或直接提出问题，我会为您提供专业、准确的解答。`

		select {
		case <-ctx.Done():
			return nil
		case responseChan <- StreamResponse{Content: response}:
			return nil
		}
	}

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

	// 执行搜索
	response, err := client.Search(ctx, &qdrant.SearchPoints{
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
	})
	if err != nil {
		return fmt.Errorf("搜索文档失败: %v", err)
	}

	// 构建上下文
	var contextBuilder strings.Builder
	docContents := make(map[string]struct {
		name     string
		contents []string
	})

	// 按文档ID组织内容
	for _, point := range response.Result {
		docID := point.Payload["doc_id"].GetStringValue()
		content := point.Payload["content"].GetStringValue()
		filename := point.Payload["filename"].GetStringValue()

		if content != "" {
			doc := docContents[docID]
			doc.name = filename
			doc.contents = append(doc.contents, content)
			docContents[docID] = doc
		}
	}

	// 发送思考提示
	select {
	case <-ctx.Done():
		return nil
	case responseChan <- StreamResponse{Content: "让我查看文档内容..."}:
		time.Sleep(300 * time.Millisecond)
	}

	// 构建文档内容
	for _, doc := range docContents {
		contextBuilder.WriteString(fmt.Sprintf("\n文档《%s》内容：\n", doc.name))
		for _, content := range doc.contents {
			// 1. 基础清理
			content = strings.TrimSpace(content)

			// 2. 移除所有特殊字符
			content = regexp.MustCompile(`[*\d\.\-:：\[\]()（）<>《》]+`).ReplaceAllString(content, "")

			// 3. 移除重复的标点符号
			content = regexp.MustCompile(`[，。？！：；、]{2,}`).ReplaceAllString(content, "。")

			// 4. 处理空白字符
			content = regexp.MustCompile(`\s+`).ReplaceAllString(content, " ")

			// 5. 只保留中文、英文和基本标点
			content = regexp.MustCompile(`[^\p{Han}\p{Latin}，。？！：；、\s]+`).ReplaceAllString(content, "")

			if strings.TrimSpace(content) != "" {
				contextBuilder.WriteString(content)
				contextBuilder.WriteString("\n")
			}
		}
		contextBuilder.WriteString("\n---\n")
	}

	// 修改提示词
	prompt := fmt.Sprintf(`你是一个专业的知识库问答助手。请基于以下文档内容回答问题：

===== 文档内容开始 =====
%s
===== 文档内容结束 =====

用户问题：%s

严格要求：
1. 必须以"根据文档《xxx》"开头
2. 只能使用文档中明确提到的信息
3. 回答要详细完整，不要过于简短
4. 如果遇到文本不连贯的情况：
   - 忽略无意义的字符
   - 只使用能理解的完整片段
   - 保持专业术语的准确性
5. 确保回答的可读性和连贯性
6. 分点列举主要功能和特点

请用清晰、专业的语言回答：`, contextBuilder.String(), question)

	// 生成回答
	answer, err := s.llm.Call(ctx, prompt)
	if err != nil {
		return fmt.Errorf("生成回答失败: %v", err)
	}

	// 清除思考提示并发送回答
	select {
	case <-ctx.Done():
		return nil
	case responseChan <- StreamResponse{Content: "\r                           \n"}:
		time.Sleep(200 * time.Millisecond)
	}

	// 分段发送回答
	paragraphs := strings.Split(answer, "\n")
	for _, p := range paragraphs {
		if strings.TrimSpace(p) == "" {
			continue
		}
		select {
		case <-ctx.Done():
			return nil
		case responseChan <- StreamResponse{Content: p + "\n"}:
			time.Sleep(300 * time.Millisecond)
		}
	}

	return nil
}

// getDocumentSummary 获取文档摘要
func (s *QAService) getDocumentSummary(ctx context.Context, documentIds []string) (string, error) {
	// 获取文档内容
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

	// 执行搜索
	response, err := client.Search(ctx, &qdrant.SearchPoints{
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
	})
	if err != nil {
		return "", fmt.Errorf("搜索文档失败: %v", err)
	}

	// 按文档ID组织内容
	docContents := make(map[string]struct {
		name     string
		contents []string
	})

	for _, point := range response.Result {
		docID := point.Payload["doc_id"].GetStringValue()
		content := point.Payload["content"].GetStringValue()
		filename := point.Payload["filename"].GetStringValue()

		if content != "" {
			doc := docContents[docID]
			doc.name = filename
			doc.contents = append(doc.contents, content)
			docContents[docID] = doc
		}
	}

	// 构建文档内容
	var contextBuilder strings.Builder
	for _, doc := range docContents {
		contextBuilder.WriteString(fmt.Sprintf("\n文档《%s》内容：\n", doc.name))
		for _, content := range doc.contents {
			// 清理内容中的特殊字符
			cleanContent := cleanText(content)
			contextBuilder.WriteString(cleanContent)
			contextBuilder.WriteString("\n")
		}
		contextBuilder.WriteString("\n---\n")
	}

	// 构建提示词
	prompt := fmt.Sprintf(`请仔细分析并总结以下文档的主要内容：

===== 文档内容开始 =====
%s
===== 文档内容结束 =====

要求：
1. 以"根据文档《xxx》，主要内容如下："的格式开始
2. 分点列出3-5个核心要点
3. 每个要点需要详细说明，并引用原文关键内容
4. 保持内容的准确性和完整性
5. 只包含文档中明确提到的信息
6. 如果是技术文档，需要突出技术重点
7. 如果是业务文档，需要突出业务关键点

请开始总结：`, contextBuilder.String())

	// 生成摘要
	summary, err := s.llm.Call(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("生成摘要失败: %v", err)
	}

	return summary, nil
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
		docID := point.Payload["doc_id"].GetStringValue()
		if docID == "" {
			continue
		}

		if _, exists := docMap[docID]; !exists {
			// 获取上传时间
			defaultTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC) // 使用固定的默认时间
			uploadTime := defaultTime

			if timeStr := point.Payload["upload_time"].GetStringValue(); timeStr != "" {
				if parsedTime, err := time.Parse(time.RFC3339, timeStr); err == nil {
					uploadTime = parsedTime
				}
			}

			docMap[docID] = &models.Document{
				ID:         docID,
				Name:       point.Payload["filename"].GetStringValue(),
				UploadTime: uploadTime,
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

// 辅助函数：提取相关句子
func extractRelevantSentences(text, keyword string) []string {
	sentences := strings.Split(text, "。")
	var relevant []string
	seenSentences := make(map[string]bool)

	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence != "" && strings.Contains(sentence, keyword) && !seenSentences[sentence] {
			seenSentences[sentence] = true
			relevant = append(relevant, sentence)
		}
	}

	return relevant
}
