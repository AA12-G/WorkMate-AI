package models

import "time"

type Document struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Type        string    `json:"type"`
	Size        int64     `json:"size"`
	UploadTime  time.Time `json:"upload_time"`
	Description string    `json:"description,omitempty"`
}

type DocumentList struct {
	Documents []Document `json:"documents"`
	Total     int        `json:"total"`
}
