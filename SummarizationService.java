package com.analyzer.aisummarizer.services;

public interface SummarizationService {
    String summarize(String text, Long transcriptionId) throws Exception;
}