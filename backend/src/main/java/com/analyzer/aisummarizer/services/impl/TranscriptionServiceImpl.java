package com.analyzer.aisummarizer.services.impl;

import com.analyzer.aisummarizer.entity.Transcription;
import com.analyzer.aisummarizer.entity.Video;
import com.analyzer.aisummarizer.repository.TranscriptionRepository;
import com.analyzer.aisummarizer.repository.VideoRepository;
import com.analyzer.aisummarizer.services.TranscriptionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.time.LocalDateTime;

@Service
public class TranscriptionServiceImpl implements TranscriptionService {

    private final VideoRepository videoRepository;
    private final TranscriptionRepository transcriptionRepository;
    private final RestTemplate restTemplate;
    private final String flaskUrl = "http://192.168.137.41:5001/transcribe";

    @Autowired
    public TranscriptionServiceImpl(VideoRepository videoRepository,
                                    TranscriptionRepository transcriptionRepository) {
        this.videoRepository = videoRepository;
        this.transcriptionRepository = transcriptionRepository;
        this.restTemplate = new RestTemplate();
    }

    @Override
    public String transcribe(MultipartFile file) throws Exception {
        // Save file temporarily
        File tempFile = File.createTempFile("audio-", file.getOriginalFilename());
        try (FileOutputStream fos = new FileOutputStream(tempFile)) {
            fos.write(file.getBytes());
        }

        // Prepare multipart form data for Flask API
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };
        body.add("file", resource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // Call Flask /transcribe endpoint
        String transcriptionText = restTemplate.postForObject(flaskUrl, requestEntity, String.class);

        // Save Video record (metadata)
        Video video = new Video();
        video.setFileName(file.getOriginalFilename());
        video.setFilePath(tempFile.getAbsolutePath());
        video.setUploadedAt(LocalDateTime.now());
        Video savedVideo = videoRepository.save(video);

        // Save Transcription record linked to the video
        Transcription transcription = new Transcription();
        transcription.setVideo(savedVideo);
        transcription.setTranscribedText(transcriptionText);
        transcriptionRepository.save(transcription);

        // Delete temporary file
        tempFile.delete();

        return transcriptionText;
    }
}