package handlers

import (
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"your-project/internal/services"

	"github.com/gin-gonic/gin"
)

type QAHandler struct {
	qaService *services.QAService
}

func NewQAHandler(qaService *services.QAService) *QAHandler {
	return &QAHandler{
		qaService: qaService,
	}
}

type QueryRequest struct {
	Question string `json:"question" binding:"required"`
}

type QueryResponse struct {
	Answer string `json:"answer"`
}

// UploadFile 处理文件上传
func (h *QAHandler) UploadFile(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 检查文件扩展名
	ext := filepath.Ext(file.Filename)
	if !isValidFileType(ext) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "不支持的文件类型"})
		return
	}

	filename := filepath.Join("file/upload", file.Filename)
	if err := c.SaveUploadedFile(file, filename); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 处理上传的文件
	if err := h.qaService.ProcessFile(c.Request.Context(), filename); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "文件上传成功"})
}

// isValidFileType 检查文件类型是否支持
func isValidFileType(ext string) bool {
	supportedTypes := map[string]bool{
		".pdf":  true,
		".txt":  true,
		".doc":  true,
		".docx": true,
		".md":   true,
		".csv":  true,
	}
	return supportedTypes[strings.ToLower(ext)]
}

// Query 处理问答请求，支持流式响应
func (h *QAHandler) Query(c *gin.Context) {
	var req QueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 设置 SSE 头部
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Transfer-Encoding", "chunked")
	c.Header("Access-Control-Allow-Origin", "*") // 允许跨域访问

	// 创建通道接收流式响应
	responseChan := make(chan string)

	// 在新的 goroutine 中处理查询
	go func() {
		if err := h.qaService.StreamingQuery(c.Request.Context(), req.Question, responseChan); err != nil {
			c.SSEvent("error", err.Error())
		}
	}()

	// 发送流式响应
	c.Stream(func(w io.Writer) bool {
		response, ok := <-responseChan
		if !ok {
			return false
		}
		c.SSEvent("message", response)
		return true
	})
}
