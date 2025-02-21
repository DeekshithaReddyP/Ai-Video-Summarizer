package com.analyzer.aisummarizer.models;

import lombok.Data;
import java.util.List;

@Data
public class KeyframeResponse {
    private List<String> keyframes;

	public List<String> getKeyframes() {
		return keyframes;
	}

	public void setKeyframes(List<String> keyframes) {
		this.keyframes = keyframes;
	}
}