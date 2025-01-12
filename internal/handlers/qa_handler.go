package handlers

import (
	"net/http"
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

func (h *QAHandler) Query(c *gin.Context) {
	var req QueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	answer, err := h.qaService.Query(c.Request.Context(), req.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, QueryResponse{Answer: answer})
}
