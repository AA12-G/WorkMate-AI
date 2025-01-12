package services

import (
	"context"
	"your-project/config"
	"your-project/pkg/vectorstore"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/vectorstores"
)

type QAService struct {
	llm    *openai.LLM
	qdrant *vectorstore.QdrantStore
	config *config.Config
}

func NewQAService(cfg *config.Config) (*QAService, error) {
	llm, err := openai.New(
		openai.WithToken(cfg.OpenAIAPIKey),
		openai.WithBaseURL(cfg.OpenAIBaseURL),
	)
	if err != nil {
		return nil, err
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		return nil, err
	}

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
		llm:    llm,
		qdrant: qdrant,
		config: cfg,
	}, nil
}

func (s *QAService) Query(ctx context.Context, question string) (string, error) {
	qa := chains.NewRetrievalQAFromLLM(
		s.llm,
		vectorstores.ToRetriever(s.qdrant.GetStore(), 10),
	)

	return chains.Run(ctx, qa, question)
}
