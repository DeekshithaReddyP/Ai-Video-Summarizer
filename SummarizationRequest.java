package com.analyzer.aisummarizer.models;

import lombok.Data;

@Data
public class SummarizationRequest {
    private String text;
    private Long transcriptionId;
	public Long getTranscriptionId() {
		return transcriptionId;
	}
	public void setTranscriptionId(Long transcriptionId) {
		this.transcriptionId = transcriptionId;
	}
	public String getText() {
		return text;
	}
	public void setText(String text) {
		this.text = text;
	}
}