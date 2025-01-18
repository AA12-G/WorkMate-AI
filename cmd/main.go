package main

import (
	"WorkMate-AI/config"
	"WorkMate-AI/internal/handlers"
	"WorkMate-AI/internal/services"
	"log"

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

	// 添加 CORS 中间件
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")

		// 处理预检请求
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	setupRoutes(r, qaHandler)

	// 启动服务器
	log.Printf("Server starting on port %s", cfg.ServerPort)
	if err := r.Run(cfg.ServerPort); err != nil {
		log.Fatal(err)
	}
}

func setupRoutes(router *gin.Engine, qaHandler *handlers.QAHandler) {
	// 添加路由日志中间件（移到最前面）
	router.Use(func(c *gin.Context) {
		log.Printf("请求路径: %s, 方法: %s", c.Request.URL.Path, c.Request.Method)
		c.Next()
	})

	// 设置静态文件路由
	router.Static("/static", "./static")

	// API 路由组
	api := router.Group("/api")
	{
		// 文件上传
		api.POST("/upload", qaHandler.HandleUpload)

		// 问答接口
		api.POST("/query", qaHandler.HandleQuery)

		// 流式问答接口
		api.POST("/streaming-query", qaHandler.HandleStreamingQuery)

		// 获取文档列表
		api.GET("/documents", qaHandler.HandleListDocuments)

		// 删除文档
		api.DELETE("/documents/:id", qaHandler.HandleDeleteDocument)
	}

	// 主页
	router.GET("/", func(c *gin.Context) {
		c.File("static/index.html")
	})
}
