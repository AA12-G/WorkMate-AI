package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"WorkMate-AI/internal/services"

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

// HandleQuery 处理问答请求
func (h *QAHandler) HandleQuery(c *gin.Context) {
	var request struct {
		Question    string   `json:"question"`
		DocumentIds []string `json:"document_ids"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求参数"})
		return
	}

	answer, err := h.qaService.Query(c.Request.Context(), request.Question, request.DocumentIds)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"answer": answer})
}

// HandleStreamingQuery 处理流式查询请求
func (h *QAHandler) HandleStreamingQuery(c *gin.Context) {
	question := c.Query("question")
	documentIdsStr := c.Query("document_ids")

	var documentIds []string
	if err := json.Unmarshal([]byte(documentIdsStr), &documentIds); err != nil {
		c.String(http.StatusBadRequest, "无效的文档ID")
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Transfer-Encoding", "chunked")

	responseChan := make(chan services.StreamResponse)
	ctx := c.Request.Context()

	go func() {
		defer close(responseChan)
		if err := h.qaService.StreamingQuery(ctx, question, documentIds, responseChan); err != nil {
			select {
			case <-ctx.Done():
			case responseChan <- services.StreamResponse{Content: fmt.Sprintf("错误：%v", err)}:
			}
		}
	}()

	c.Stream(func(w io.Writer) bool {
		select {
		case <-ctx.Done():
			return false
		case response, ok := <-responseChan:
			if !ok {
				fmt.Fprintf(w, "event: complete\ndata: \n\n")
				return false
			}
			fmt.Fprintf(w, "data: %s\n\n", response.Content)
			return true
		}
	})
}

// HandleUpload 处理文件上传
func (h *QAHandler) HandleUpload(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无法获取上传的文件"})
		return
	}

	// 创建上传目录
	uploadDir := "file/upload"
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建上传目录失败"})
		return
	}

	// 保存文件
	filePath := filepath.Join(uploadDir, file.Filename)
	if err := c.SaveUploadedFile(file, filePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "保存文件失败"})
		return
	}

	// 处理文件
	if err := h.qaService.ProcessFile(c.Request.Context(), filePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("处理文件失败: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "文件上传并处理成功"})
}

// HandleListDocuments 处理获取文档列表的请求
func (h *QAHandler) HandleListDocuments(c *gin.Context) {
	documents, err := h.qaService.ListDocuments(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, documents)
}

// HandleDeleteDocument 处理文档删除请求
func (h *QAHandler) HandleDeleteDocument(c *gin.Context) {
	docID := c.Param("id")
	log.Printf("收到删除请求，文档ID: %s", docID)

	if docID == "" {
		log.Printf("文档ID为空")
		c.JSON(http.StatusBadRequest, gin.H{"error": "文档ID不能为空"})
		return
	}

	err := h.qaService.DeleteDocument(c.Request.Context(), docID)
	if err != nil {
		log.Printf("删除文档失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("删除文档失败: %v", err)})
		return
	}

	log.Printf("文档删除成功: %s", docID)
	c.JSON(http.StatusOK, gin.H{"message": "文档删除成功"})
}
