package com.a303.missionms.domain.missionLog.service;

import com.a303.missionms.domain.dailyMission.DailyMission;
import com.a303.missionms.domain.dailyMission.repository.DailyMissionRepository;
import com.a303.missionms.domain.mission.dto.response.CalenderDayMissionRes;
import com.a303.missionms.domain.mission.dto.response.MissionLogBaseRes;
import com.a303.missionms.domain.mission.dto.response.MissionReportRes;
import com.a303.missionms.domain.mission.dto.response.MyTableMissionRes;
import com.a303.missionms.domain.missionLog.repository.MissionLogRepository;
import jakarta.transaction.Transactional;
import java.time.LocalDate;
import java.util.ArrayList;
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
public class MissionLogServiceImpl implements MissionLogService {

	private final MissionLogRepository missionLogRepository;
	private final DailyMissionRepository dailyMissionRepository;


	@Override
	public MissionReportRes getMissionReport(int memberId, int year, int month) {

		Map<Integer, List<MyTableMissionRes>> dayMap = new HashMap<>();
		Map<LocalDate, CalenderDayMissionRes> calenderDayMissionResMap = new HashMap<>();
		int[] daySuccessCountArray = new int[33];

		int percent = 0;

		// 만약 해당 월이 이번 달이면 오늘꺼도 추가해야한다.
		LocalDate now = LocalDate.now();
//		System.out.println(now.getYear()+" "+now.getMonthValue());
		if (year == now.getYear() && month == now.getMonthValue()) {
//			System.out.println(11);
			int day = now.getDayOfMonth();
			List<DailyMission> dailyMissions = dailyMissionRepository.findByMemberId(memberId);
			for (DailyMission dailyMission : dailyMissions) {
				if (dailyMission.isFinish()) {
					percent++;
					daySuccessCountArray[day] += 1;
				}

				MyTableMissionRes myTableMissionRes = MyTableMissionRes.builder()
					.missionId(dailyMission.getMission().getMissionId())
					.name(dailyMission.getMission().getName())
					.isFinish(dailyMission.isFinish())
					.build();
				if (dayMap.containsKey(day)) {
					dayMap.get(day).add(myTableMissionRes);
				} else {
					List<MyTableMissionRes> temp = new ArrayList<>();
					temp.add(myTableMissionRes);
					dayMap.put(day, temp);
				}
			}

		}

		List<MissionLogBaseRes> missionLogBaseResList = missionLogRepository.getMyReport(memberId,
			year, month);
		for (MissionLogBaseRes missionLog : missionLogBaseResList) {

			int day = missionLog.missionDate().getDayOfMonth();

			if (missionLog.isFinish()) {
				percent++;
				daySuccessCountArray[day] += 1;
			}

			MyTableMissionRes myTableMissionRes = MyTableMissionRes.builder()
				.missionId(missionLog.missionBaseRes().missionId())
				.name(missionLog.missionBaseRes().name())
				.isFinish(missionLog.isFinish())
				.build();
			if (dayMap.containsKey(day)) {
				dayMap.get(day).add(myTableMissionRes);
			} else {
				List<MyTableMissionRes> temp = new ArrayList<>();
				temp.add(myTableMissionRes);
				dayMap.put(day, temp);
			}
		}
		percent *= 100;
		if (missionLogBaseResList.size() != 0) {
			percent /= missionLogBaseResList.size();
		} else {
			percent = 0;
		}

		for (Entry<Integer, List<MyTableMissionRes>> entry : dayMap.entrySet()) {
			int day = entry.getKey();
			LocalDate dayString = LocalDate.of(year, month, day);

			CalenderDayMissionRes calenderDayMissionRes = CalenderDayMissionRes.builder()
				.count(daySuccessCountArray[day])
				.missions(entry.getValue())
				.build();

			calenderDayMissionResMap.put(dayString, calenderDayMissionRes);
		}

//		System.out.println(calenderDayMissionResMap);

		return MissionReportRes.builder()
			.percent(percent)
			.year(year)
			.month(month)
			.missionCalender(calenderDayMissionResMap)
			.build();
	}

	@Override
	public long getCompletedMissionCount(int memberId) {

		long logCnt = missionLogRepository.countByMemberIdAndIsFinishIsTrue(memberId);
		long dayCnt = dailyMissionRepository.countByMemberIdAndIsFinishIsTrue(memberId);

		return logCnt + dayCnt;
	}
}
