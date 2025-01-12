package config

import (
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	OpenAIAPIKey   string
	OpenAIBaseURL  string
	QdrantGRPCURL  string
	QdrantHTTPURL  string
	CollectionName string
	ServerPort     string
}

func NewConfig() *Config {
	// 加载 .env 文件
	_ = godotenv.Load()

	return &Config{
		OpenAIAPIKey:   getEnv("OPENAI_API_KEY", ""),
		OpenAIBaseURL:  getEnv("OPENAI_BASE_URL", "https://api.aiproxy.io/v1"),
		QdrantGRPCURL:  getEnv("QDRANT_GRPC_URL", "localhost:6334"),
		QdrantHTTPURL:  getEnv("QDRANT_HTTP_URL", "http://localhost:6333"),
		CollectionName: getEnv("COLLECTION_NAME", "knowledge-base"),
		ServerPort:     getEnv("SERVER_PORT", ":8080"),
	}
}

// getEnv 获取环境变量，如果不存在则返回默认值
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
