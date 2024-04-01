package com.a303.missionms.domain.dailyMission.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Slf4j
public class DailyMissionScheduler {

	private final DailyMissionService dailyMissionService;

	private final int ONE_MINUTE = 60000;

	@Scheduled(cron = "5 0 12 * * *", zone = "Asia/Seoul")  // 매일 12시 00분 5초
	@Transactional
	public void dailyMissionRecommend() {
		dailyMissionService.dailyMissionRecommend();
	}

}