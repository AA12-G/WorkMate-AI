package main

import (
	"log"
	"your-project/config"
	"your-project/internal/handlers"
	"your-project/internal/services"

	"github.com/gin-gonic/gin"
)

func main() {
	// 加载配置
	cfg := config.NewConfig()

	// 初始化服务
	qaService, err := services.NewQAService(cfg)
	if err != nil {
		log.Fatal(err)
	}

	// 初始化处理器
	qaHandler := handlers.NewQAHandler(qaService)

	// 设置路由
	r := gin.Default()

	// 添加中间件
	r.Use(gin.Recovery())
	r.Use(gin.Logger())

	// 静态文件服务
	r.Static("/uploads", "./file/upload")

	// API路由
	api := r.Group("/api")
	{
		api.POST("/upload", qaHandler.UploadFile)
		api.POST("/query", qaHandler.Query)
	}

	// 启动服务器
	log.Printf("Server starting on port %s", cfg.ServerPort)
	if err := r.Run(cfg.ServerPort); err != nil {
		log.Fatal(err)
	}
}
