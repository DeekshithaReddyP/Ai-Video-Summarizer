package com.analyzer.aisummarizer.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "keyframes")
public class Keyframe {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "video_id", referencedColumnName = "id")
    private Video video;

    private String keyframePath;
    private Integer frameNumber;
	public String getKeyframePath() {
		return keyframePath;
	}
	public void setKeyframePath(String keyframePath) {
		this.keyframePath = keyframePath;
	}
	public Integer getFrameNumber() {
		return frameNumber;
	}
	public void setFrameNumber(Integer frameNumber) {
		this.frameNumber = frameNumber;
	}
	public Long getId() {
		return id;
	}
	public void setId(Long id) {
		this.id = id;
	}
	public Video getVideo() {
		return video;
	}
	public void setVideo(Video video) {
		this.video = video;
	}
	
}