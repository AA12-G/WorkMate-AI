package main

import (
	"context"
	"log"
	"net/url"
	"os"
	"your-project/config"

	qdrantapi "github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// 加载配置
	cfg := config.NewConfig()

	ctx := context.Background()

	// 连接到 Qdrant
	conn, err := grpc.Dial(cfg.QdrantGRPCURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	// 创建 Qdrant 客户端
	collections := qdrantapi.NewCollectionsClient(conn)

	// 创建集合
	_, err = collections.Create(ctx, &qdrantapi.CreateCollection{
		CollectionName: cfg.CollectionName,
		VectorsConfig: &qdrantapi.VectorsConfig{
			Config: &qdrantapi.VectorsConfig_Params{
				Params: &qdrantapi.VectorParams{
					Size:     1536,
					Distance: qdrantapi.Distance_Cosine,
				},
			},
		},
	})
	if err != nil {
		log.Printf("Create collection error (might already exist): %v", err)
	}

	// 打开并读取PDF文件
	f, err := os.Open("./file/upload/test.pdf")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	finfo, err := f.Stat()
	if err != nil {
		log.Fatal(err)
	}

	// 加载PDF文档
	p := documentloaders.NewPDF(f, finfo.Size())

	// 分割文档
	chunkDocuments, err := p.LoadAndSplit(ctx, textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(2000),
		textsplitter.WithChunkOverlap(200),
	))
	if err != nil {
		log.Fatal(err)
	}

	// 初始化 OpenAI 客户端
	llm, err := openai.New(
		openai.WithToken(cfg.OpenAIAPIKey),
		openai.WithBaseURL(cfg.OpenAIBaseURL),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 创建 embedder
	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatal(err)
	}

	// 解析 Qdrant HTTP URL
	qdrantHTTPURL, err := url.Parse(cfg.QdrantHTTPURL)
	if err != nil {
		log.Fatal(err)
	}

	// 初始化 Qdrant 向量存储
	store, err := qdrant.New(
		qdrant.WithURL(*qdrantHTTPURL),
		qdrant.WithEmbedder(embedder),
		qdrant.WithCollectionName(cfg.CollectionName),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 添加文档到向量存储
	docsIndex, err := store.AddDocuments(ctx, chunkDocuments)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Documents indexed:", docsIndex)

	// 发起提问
	qa := chains.NewRetrievalQAFromLLM(llm, vectorstores.ToRetriever(store, 10))
	res, err := chains.Run(ctx, qa, "请详细列出简历中所有的工作经历，包括所有公司名称和工作时间")
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Query result:", res)
}
