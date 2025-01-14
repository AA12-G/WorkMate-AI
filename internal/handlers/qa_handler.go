package handlers

import (
	"WorkMate-AI/internal/models"
	"WorkMate-AI/internal/services"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"

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
	Answer   string           `json:"answer"`
	Document *models.Document `json:"document,omitempty"`
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
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 设置 SSE 头部
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Transfer-Encoding", "chunked")
	c.Header("Access-Control-Allow-Origin", "*")

	responseChan := make(chan services.StreamResponse)

	go func() {
		if err := h.qaService.StreamingQuery(c.Request.Context(), req.Question, responseChan); err != nil {
			c.SSEvent("error", err.Error())
		}
	}()

	c.Stream(func(w io.Writer) bool {
		response, ok := <-responseChan
		if !ok {
			return false
		}

		// 直接写入纯文本数据
		fmt.Fprintf(w, "%s", string(response))
		return true
	})
}

func (h *QAHandler) ListDocuments(c *gin.Context) {
	documents, err := h.qaService.ListDocuments(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "获取文档列表失败: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, documents)
}
