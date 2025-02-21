package com.analyzer.aisummarizer.services;

import com.analyzer.aisummarizer.models.KeyframeResponse;
import org.springframework.web.multipart.MultipartFile;

public interface KeyframeService {
    KeyframeResponse extractKeyframes(MultipartFile file, Long videoId) throws Exception;
}