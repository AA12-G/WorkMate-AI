package services

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path"
	"strings"
	"time"
	"your-project/config"
	"your-project/pkg/vectorstore"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
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

	qdrant, err := vectorstore.NewQdrantStore(
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
		qdrant: qdrant,
		config: cfg,
	}, nil
}

// ProcessFile 处理上传的文件
func (s *QAService) ProcessFile(ctx context.Context, filepath string) error {
	// 打开文件
	f, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	finfo, err := f.Stat()
	if err != nil {
		return err
	}

	// 根据文件类型选择合适的加载器
	var loader documentloaders.Loader
	ext := strings.ToLower(path.Ext(filepath))

	switch ext {
	case ".pdf":
		loader = documentloaders.NewPDF(f, finfo.Size())
	case ".txt", ".md":
		loader = documentloaders.NewText(f)
	case ".csv":
		loader = documentloaders.NewCSV(f)
	case ".doc", ".docx":
		// 需要额外的库支持 Word 文档
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
	return err
}

func (s *QAService) Query(ctx context.Context, question string) (string, error) {
	qa := chains.NewRetrievalQAFromLLM(
		s.llm,
		vectorstores.ToRetriever(s.qdrant.GetStore(), 10),
	)

	return chains.Run(ctx, qa, question)
}

// StreamingQuery 提供流式问答服务
func (s *QAService) StreamingQuery(ctx context.Context, question string, responseChan chan string) error {
	qa := chains.NewRetrievalQAFromLLM(
		s.llm,
		vectorstores.ToRetriever(s.qdrant.GetStore(), 10),
	)

	// 使用 goroutine 处理查询
	go func() {
		// 获取完整答案
		answer, err := chains.Run(ctx, qa, question)
		if err != nil {
			select {
			case <-ctx.Done():
			case responseChan <- "Error: " + err.Error():
			}
			close(responseChan)
			return
		}

		// 按字符发送，实现打字机效果
		runes := []rune(answer)
		for i := 0; i < len(runes); i++ {
			select {
			case <-ctx.Done():
				close(responseChan)
				return
			case responseChan <- string(runes[i : i+1]):
				time.Sleep(80 * time.Millisecond) // 控制打字速度
				// 继续发送
			}
		}
		close(responseChan)
	}()

	return nil
}
