package com.analyzer.aisummarizer.controllers;

import com.analyzer.aisummarizer.models.KeyframeResponse;
import com.analyzer.aisummarizer.models.SummarizationRequest;
import com.analyzer.aisummarizer.services.KeyframeService;
import com.analyzer.aisummarizer.services.SummarizationService;
import com.analyzer.aisummarizer.services.TranscriptionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class AIController {

    @Autowired
    private TranscriptionService transcriptionService;

    @Autowired
    private SummarizationService summarizationService;

    @Autowired
    private KeyframeService keyframeService;

//     Endpoint to perform speech-to-text transcription
    @PostMapping("/transcribe")
    public ResponseEntity<String> transcribe(@RequestParam("file") MultipartFile file) {
        try {
            String transcription = transcriptionService.transcribe(file);
            return ResponseEntity.ok(transcription);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error during transcription: " + e.getMessage());
        }
    }

    // Endpoint to perform text summarization
    @PostMapping("/summarize")
    public ResponseEntity<String> summarize(@RequestBody SummarizationRequest request) {
        try {
            String summary = summarizationService.summarize(request.getText(), request.getTranscriptionId());
            return ResponseEntity.ok(summary);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error during summarization: " + e.getMessage());
        }
    }

    // Endpoint to extract keyframes from video	
    @PostMapping("/keyframes")
    public ResponseEntity<KeyframeResponse> extractKeyframes(@RequestParam("file") MultipartFile file,
                                                             @RequestParam("videoId") Long videoId) {
        try {
            KeyframeResponse response = keyframeService.extractKeyframes(file, videoId);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(null);
        }
    }
    @GetMapping("/keyframes")
    public ResponseEntity<String> getKeyframes() {
        // return keyframes data
        return ResponseEntity.ok("Keyframes API is working!");
    }
}