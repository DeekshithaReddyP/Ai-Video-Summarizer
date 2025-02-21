package com.analyzer.aisummarizer.repository;

import com.analyzer.aisummarizer.entity.Keyframe;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface KeyframeRepository extends JpaRepository<Keyframe, Long> {
}