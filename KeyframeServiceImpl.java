package com.analyzer.aisummarizer.services.impl;

import com.analyzer.aisummarizer.entity.Keyframe;
import com.analyzer.aisummarizer.entity.Video;
import com.analyzer.aisummarizer.models.KeyframeResponse;
import com.analyzer.aisummarizer.repository.KeyframeRepository;
import com.analyzer.aisummarizer.repository.VideoRepository;
import com.analyzer.aisummarizer.services.KeyframeService;
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
import java.util.ArrayList;
import java.util.List;

@Service
public class KeyframeServiceImpl implements KeyframeService {

    private final VideoRepository videoRepository;
    private final KeyframeRepository keyframeRepository;
    private final RestTemplate restTemplate;
    private final String flaskUrl = "http://192.168.137.41:5001/keyframes";

    @Autowired
    public KeyframeServiceImpl(VideoRepository videoRepository,
                               KeyframeRepository keyframeRepository) {
        this.videoRepository = videoRepository;
        this.keyframeRepository = keyframeRepository;
        this.restTemplate = new RestTemplate();
    }

    @Override
    public KeyframeResponse extractKeyframes(MultipartFile file, Long videoId) throws Exception {
        // Save video file temporarily
        File tempFile = File.createTempFile("video-", file.getOriginalFilename());
        try (FileOutputStream fos = new FileOutputStream(tempFile)) {
            fos.write(file.getBytes());
        }

        // Prepare multipart form data for Flask API
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        ByteArrayResource resource = new ByteArrayResource(file.getBytes()){
            @Override
            public String getFilename(){
                return file.getOriginalFilename();
            }
        };
        body.add("file", resource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // Call Flask /keyframes endpoint
        String responseJson = restTemplate.postForObject(flaskUrl, requestEntity, String.class);

        // Naively parse the JSON response (in production, use a JSON parser)
        List<String> keyframePaths = new ArrayList<>();
        if (responseJson != null && responseJson.contains("keyframes")) {
            int start = responseJson.indexOf('[');
            int end = responseJson.indexOf(']');
            if (start != -1 && end != -1) {
                String listStr = responseJson.substring(start + 1, end);
                String[] paths = listStr.split(",");
                for (String path : paths) {
                    keyframePaths.add(path.trim().replace("\"", ""));
                }
            }
        }

        // Save each keyframe record linked to the video
        Video video = videoRepository.findById(videoId)
                .orElseThrow(() -> new Exception("Video not found"));

        int frameNumber = 0;
        for (String path : keyframePaths) {
            Keyframe keyframe = new Keyframe();
            keyframe.setVideo(video);
            keyframe.setKeyframePath(path);
            keyframe.setFrameNumber(frameNumber++);
            keyframeRepository.save(keyframe);
        }

        // Delete temporary file
        tempFile.delete();

        KeyframeResponse response = new KeyframeResponse();
        response.setKeyframes(keyframePaths);
        return response;
    }
}