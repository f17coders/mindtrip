package com.a303.missionms.domain.mission.service;

import com.a303.missionms.domain.mission.Category;
import com.a303.missionms.domain.mission.Mission;
import com.a303.missionms.domain.mission.dto.response.CategoryMissionRes;
import com.a303.missionms.domain.mission.dto.response.MissionBaseRes;
import com.a303.missionms.domain.mission.dto.response.MissionListRes;
import com.a303.missionms.domain.mission.repository.MissionRepository;
import com.a303.missionms.global.client.MemberClient;
import com.a303.missionms.global.exception.BaseExceptionHandler;
import com.a303.missionms.global.exception.code.ErrorCode;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import jakarta.transaction.Transactional;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class MissionServiceImpl implements MissionService {

	private final MissionRepository missionRepository;

	@Override
	public MissionListRes getMissionList() throws BaseExceptionHandler, IOException {

		Map<Category, List> missionListMap = new HashMap<>();

		List<Mission> missions = missionRepository.findAll();
		for (Mission mission : missions) {
			MissionBaseRes missionBaseRes = MissionBaseRes.builder()
				.missionId(mission.getMissionId())
				.name(mission.getName())
				.category(mission.getCategory())
				.build();

			if (!missionListMap.containsKey(missionBaseRes.category())) {
				ArrayList<MissionBaseRes> temp = new ArrayList<>();
				temp.add(missionBaseRes);
				missionListMap.put(missionBaseRes.category(),
					temp);
			} else {
				missionListMap.get(missionBaseRes.category())
					.add(missionBaseRes);
			}
		}

		List<CategoryMissionRes> categoryMissionResList = new ArrayList<>();
		for (Entry<Category, List> entry : missionListMap.entrySet()) {
			categoryMissionResList.add(CategoryMissionRes.builder()
				.category(entry.getKey())
				.missionBaseResList(entry.getValue())
				.build());
		}

		return MissionListRes.builder()
			.categoryMissionResList(categoryMissionResList)
			.build();
	}

	@Override
	@CircuitBreaker(name = "customCircuitBreaker", fallbackMethod = "callFallback")
	public void welcome() throws BaseExceptionHandler {
		long before = System.currentTimeMillis();
		try {
			Thread.sleep(5000L);
		} catch (InterruptedException e) {
			throw new BaseExceptionHandler(ErrorCode.BAD_REQUEST_ERROR);
		}
		long after = System.currentTimeMillis();
		log.info("[MissionService] call => {}ms", after - before);

	}

	public String callFallback(Exception e) {
		log.error("[MissionService] callFallback");
		return "fallback";
	}
}
