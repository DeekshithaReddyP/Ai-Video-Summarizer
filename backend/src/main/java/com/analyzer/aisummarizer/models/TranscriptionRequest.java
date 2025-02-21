package com.analyzer.aisummarizer.models;

import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

@Data
public class TranscriptionRequest {
    private MultipartFile file;

	public MultipartFile getFile() {
		return file;
	}

	public void setFile(MultipartFile file) {
		this.file = file;
	}
}