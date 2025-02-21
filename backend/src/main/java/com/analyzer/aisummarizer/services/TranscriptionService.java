package com.analyzer.aisummarizer.services;

import org.springframework.web.multipart.MultipartFile;

public interface TranscriptionService {
    String transcribe(MultipartFile file) throws Exception;
}