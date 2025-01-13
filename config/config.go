package config

import (
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	QdrantGRPCURL              string
	QdrantHTTPURL              string
	CollectionName             string
	ServerPort                 string
	UseOllama                  bool
	OllamaModel                string
	DocumentMetadataCollection string
}

func NewConfig() *Config {
	// 加载 .env 文件
	_ = godotenv.Load()

	return &Config{
		QdrantGRPCURL:              getEnv("QDRANT_GRPC_URL", "localhost:6334"),
		QdrantHTTPURL:              getEnv("QDRANT_HTTP_URL", "http://localhost:6333"),
		CollectionName:             getEnv("COLLECTION_NAME", "knowledge-base"),
		ServerPort:                 getEnv("SERVER_PORT", ":8080"),
		UseOllama:                  getEnvBool("USE_OLLAMA", true),
		OllamaModel:                getEnv("OLLAMA_MODEL", "llama3.2"),
		DocumentMetadataCollection: getEnv("DOC_METADATA_COLLECTION", "document-metadata"),
	}
}

// getEnv 获取环境变量，如果不存在则返回默认值
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return value == "true"
	}
	return defaultValue
}
