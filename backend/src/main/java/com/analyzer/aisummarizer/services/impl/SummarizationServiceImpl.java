package com.analyzer.aisummarizer.services.impl;

import com.analyzer.aisummarizer.entity.Summary;
import com.analyzer.aisummarizer.entity.Transcription;
import com.analyzer.aisummarizer.repository.SummaryRepository;
import com.analyzer.aisummarizer.repository.TranscriptionRepository;
import com.analyzer.aisummarizer.services.SummarizationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class SummarizationServiceImpl implements SummarizationService {

    private final TranscriptionRepository transcriptionRepository;
    private final SummaryRepository summaryRepository;
    private final RestTemplate restTemplate;
    private final String flaskUrl = "http://192.168.137.41:5001/summarize";

    @Autowired
    public SummarizationServiceImpl(TranscriptionRepository transcriptionRepository,
                                    SummaryRepository summaryRepository) {
        this.transcriptionRepository = transcriptionRepository;
        this.summaryRepository = summaryRepository;
        this.restTemplate = new RestTemplate();
    }

    @Override
    public String summarize(String text, Long transcriptionId) throws Exception {
        // Prepare JSON request body
        String requestJson = "{\"text\":\"" + text.replace("\"", "\\\"") + "\"}";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> requestEntity = new HttpEntity<>(requestJson, headers);

        // Call Flask /summarize endpoint
        String summaryText = restTemplate.postForObject(flaskUrl, requestEntity, String.class);

        // Retrieve Transcription record
        Transcription transcription = transcriptionRepository.findById(transcriptionId)
                .orElseThrow(() -> new Exception("Transcription not found"));

        // Save Summary record linked to the transcription
        Summary summary = new Summary();
        summary.setTranscription(transcription);
        summary.setSummaryText(summaryText);
        summaryRepository.save(summary);

        return summaryText;
    }
}