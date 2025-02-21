package com.analyzer.aisummarizer.repository;

import com.analyzer.aisummarizer.entity.Transcription;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TranscriptionRepository extends JpaRepository<Transcription, Long> {
}