package vectorstore

import (
	"context"
	"net/url"

	qdrantapi "github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type QdrantStore struct {
	store vectorstores.VectorStore
	conn  *grpc.ClientConn
}

func NewQdrantStore(ctx context.Context, grpcURL, httpURL, collectionName string, embedder embeddings.Embedder) (*QdrantStore, error) {
	conn, err := grpc.Dial(grpcURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	collections := qdrantapi.NewCollectionsClient(conn)

	// 创建集合
	_, err = collections.Create(ctx, &qdrantapi.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: &qdrantapi.VectorsConfig{
			Config: &qdrantapi.VectorsConfig_Params{
				Params: &qdrantapi.VectorParams{
					Size:     3072,
					Distance: qdrantapi.Distance_Cosine,
				},
			},
		},
	})
	if err != nil {
		// 忽略已存在的集合错误
	}

	qdrantURL, err := url.Parse(httpURL)
	if err != nil {
		return nil, err
	}

	store, err := qdrant.New(
		qdrant.WithURL(*qdrantURL),
		qdrant.WithEmbedder(embedder),
		qdrant.WithCollectionName(collectionName),
	)
	if err != nil {
		return nil, err
	}

	return &QdrantStore{
		store: store,
		conn:  conn,
	}, nil
}

func (qs *QdrantStore) Close() {
	if qs.conn != nil {
		qs.conn.Close()
	}
}

func (qs *QdrantStore) GetStore() vectorstores.VectorStore {
	return qs.store
}
