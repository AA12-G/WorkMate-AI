package main

import (
	"log"
	"net/http"
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

	// 添加 CORS 中间件
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "*")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// 静态文件服务
	r.Static("/static", "./static")
	r.Static("/uploads", "./file/upload")

	// 处理 favicon.ico 请求
	r.GET("/favicon.ico", func(c *gin.Context) {
		c.Status(http.StatusNoContent)
	})

	// 添加根路由重定向到 index.html
	r.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/static/index.html")
	})

	// API路由
	api := r.Group("/api")
	{
		api.POST("/upload", qaHandler.UploadFile)
		api.POST("/query", qaHandler.Query)
		api.GET("/documents", qaHandler.ListDocuments)
	}

	// 启动服务器
	log.Printf("Server starting on port %s", cfg.ServerPort)
	if err := r.Run(cfg.ServerPort); err != nil {
		log.Fatal(err)
	}
}
