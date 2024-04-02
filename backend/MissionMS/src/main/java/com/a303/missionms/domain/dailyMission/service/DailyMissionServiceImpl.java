package com.a303.missionms.domain.dailyMission.service;

import com.a303.missionms.domain.dailyMission.DailyMission;
import com.a303.missionms.domain.dailyMission.dto.NotificationEventDto;
import com.a303.missionms.domain.dailyMission.repository.DailyMissionRepository;
import com.a303.missionms.domain.mission.Mission;
import com.a303.missionms.domain.mission.dto.request.MyTableMissionDTO;
import com.a303.missionms.domain.mission.dto.response.DailyMissionBaseRes;
import com.a303.missionms.domain.mission.dto.response.MissionLogRes;
import com.a303.missionms.domain.mission.dto.response.MyTableMissionRes;
import com.a303.missionms.domain.mission.repository.MissionBulkRepository;
import com.a303.missionms.domain.mission.repository.MissionRepository;
import com.a303.missionms.domain.missionLog.repository.MissionLogRepository;
import com.a303.missionms.global.api.response.BaseResponse;
import com.a303.missionms.global.client.MemberClient;
import com.a303.missionms.global.client.NotificationClient;
import com.a303.missionms.global.exception.BaseExceptionHandler;
import com.a303.missionms.global.exception.code.ErrorCode;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.EntityManager;
import jakarta.persistence.criteria.CriteriaBuilder.In;
import jakarta.transaction.Transactional;
import java.io.IOException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class DailyMissionServiceImpl implements DailyMissionService {

	@Autowired
	private final EntityManager em;

	private final KafkaTemplate<String, String> notificationEventDtoKafkaTemplate;

	private final DailyMissionRepository dailyMissionRepository;
	private final MissionLogRepository missionLogRepository;
	private final MissionRepository missionRepository;
	private final MissionBulkRepository missionBulkRepository;
	private final MemberClient memberClient;
	private final NotificationClient notificationClient;


	// TODO missionId가 바뀌지는 않지만 만약 없는 미션아이디일 경우의 예외처리도 필요하다.
	@Override
	public List<MyTableMissionRes> putMyTableMissions(int memberId,
		HashMap<Integer, MyTableMissionDTO> myTableMissionDTOMap)
		throws BaseExceptionHandler, IOException {

//		for (Entry<Integer, MyTableMissionDTO> m : myTableMissionDTOMap.entrySet()) {
//			log.error(m.toString());
//		}

		// 스케쥴링 시간과 겹치지는 않는지 체크
		if (!isValidRequestTime()) {
			log.error("putMyTableMissions method failed with request-time:{}", LocalDateTime.now());

			throw new BaseExceptionHandler("12시~12시5분은 수정,삭제,추가 요청이 불가합니다.",
				ErrorCode.INVALID_REQUEST_TIME_EXCEPTION);
		}

		// 기존 저장된 유저의 마이테이블 미션들을 엔티티 형태로 가져온다. -> n+1 해결
		List<DailyMission> dailyMissions = dailyMissionRepository.findByMemberId(
			memberId);

		// 기존 마이테이블 미션들 중에 수행된것 제외, 넣으려는 것과 동일한 것 제외하고 남은 애들은 이번에 교체되는 애들
		// 따라서 해당 엔티티는 삭제한다.
		for (int i = dailyMissions.size() - 1; i >= 0; i--) {
			DailyMission temp = dailyMissions.get(i);

			// 기존 마이테이블 미션들 중에 수행된것 제외, 넣으려는 것과 동일한 것 제외하고
			if (temp.isFinish() || myTableMissionDTOMap.containsKey(
				temp.getMission().getMissionId())) {
				dailyMissions.remove(i);
				myTableMissionDTOMap.remove(temp.getMission().getMissionId());
			}
		}

		// 남은 애들은 이번에 교체되는 애들. 따라서 해당 엔티티는 삭제한다.
		if (dailyMissions.size() != 0) {
			dailyMissionRepository.deleteAll(dailyMissions);
		}
		// 남은 애들은 이번에 들어가는 애들이다. 따라서 해당 객체는 엔티티화한다.
		List<DailyMissionBaseRes> missionsToInsert = new ArrayList<>();
		if (myTableMissionDTOMap.size() != 0) {
			List<Integer> missionIdList = new ArrayList<>(myTableMissionDTOMap.keySet());
			log.error("missionIdList : {}", missionIdList);
			List<Mission> missions = new ArrayList<>();
			for (int missionId : missionIdList) {
				Mission mission = missionRepository.findMissionByMissionId(missionId);
				missions.add(mission);
			}
//			List<Mission> missions = missionRepository.getMissionsByMissionIdIn(
//				missionIdList);
			Map<Integer, Mission> missionHashMap = new HashMap<>();
			for (Mission mission : missions) {
				missionHashMap.put(mission.getMissionId(), mission);
			}

			for (Entry<Integer, MyTableMissionDTO> entry : myTableMissionDTOMap.entrySet()) {
				Mission mission = missionHashMap.get(entry.getKey());

				DailyMissionBaseRes dailyMissionBaseRes = DailyMissionBaseRes.builder()
					.missionId(mission.getMissionId())
					.memberId(memberId)
					.isFinish(entry.getValue().isFinish())
					.build();

				missionsToInsert.add(dailyMissionBaseRes);
			}
		}
		missionBulkRepository.saveAllDailyMission(missionsToInsert);

		// 반환값은 업데이트 이후 마이테이블
		dailyMissions = dailyMissionRepository.findByMemberId(
			memberId);

		List<MyTableMissionRes> myTableMissionDTOList = new ArrayList<>();
		for (DailyMission dailyMission : dailyMissions) {
			myTableMissionDTOList.add(
				MyTableMissionRes.builder()
					.missionId(dailyMission.getMission().getMissionId())
					.name(dailyMission.getMission().getName())
					.isFinish(dailyMission.isFinish())
					.build()
			);
		}

		return myTableMissionDTOList;
	}

	@Override
	public List<MyTableMissionRes> getMyTableMissions(int memberId)
		throws BaseExceptionHandler, IOException {

//		BaseResponse<MemberBaseRes> memberBaseRes = memberClient.getMemberDtoByMemberId(memberId);
//		System.out.println(memberBaseRes.getResult());

		List<DailyMission> myTableList = dailyMissionRepository.findByMemberId(memberId);

		List<MyTableMissionRes> myTableMissionDTOList = new ArrayList<>();
		for (DailyMission dailyMission : myTableList) {
			myTableMissionDTOList.add(
				MyTableMissionRes.builder()
					.missionId(dailyMission.getMission().getMissionId())
					.name(dailyMission.getMission().getName())
					.isFinish(dailyMission.isFinish())
					.build()
			);
		}

		return myTableMissionDTOList;
	}

	@Override
	public int completeMission(int memberId, int missionId)
		throws BaseExceptionHandler, IOException {

		// 스케쥴링 시간과 겹치지는 않는지 체크
		if (!isValidRequestTime()) {
			log.error("putMyTableMissions method failed with request-time:{}", LocalDateTime.now());

			throw new BaseExceptionHandler("12시~12시5분은 수정,삭제,추가 요청이 불가합니다.",
				ErrorCode.INVALID_REQUEST_TIME_EXCEPTION);
		}

		DailyMission dailyMission = dailyMissionRepository.findByMemberIdAndMission_MissionId(
			memberId, missionId).orElseThrow(
			() -> {
				log.error("completeMission method memberId:{} missionId:{} 의 레코드가 없습니다.",
					memberId, missionId);

				return new BaseExceptionHandler(
					"해당 멤버:" + memberId + " mission: " + missionId + "의 레코드가 없습니다.",
					ErrorCode.NOT_FOUND_ERROR);
			}
		);

		log.debug("completeMission method memberId:{} missionId:{} success.", memberId, missionId);

		dailyMission.setFinish(true);

		// 수행한 미션 개수 하나 up
		BaseResponse response = memberClient.increaseMissionCount(memberId);

		return dailyMission.getDailyMissionId();
	}


	// TODO 최적화 완료
	@Override
	public void dailyMissionRecommend() throws BaseExceptionHandler {
		// daily_mission테이블 missionlog에 append
//		List<MissionLogRes> missionLogList = dailyMissionRepository.findAllToMissionLogRes();
//
//		// 미션 수행도 집계
//		float percent = 0;
//		LocalDate localDate = LocalDate.now().minusDays(1);
//		for (MissionLogRes missionLogRes : missionLogList) {
//			if (missionLogRes.isFinish()) {
//				percent++;
//			}
//		}
//		if (missionLogList.size() != 0) {
//			percent *= 100;
//			percent /= missionLogList.size();
//		}
//		log.info("미션수행도 날짜:{}, percent:{}", localDate, percent);
//
//		// jdbctemplate으로 교체(삭제와 삽입)
//		missionBulkRepository.saveAllMissionLog(missionLogList);
//		missionBulkRepository.deleteAllDailyMission();
//
//		// 새로운 미션 3개씩 선정해서 넣기
//		List<Integer> memberIdList = memberClient.getMemberIdList().getResult();
//		if (memberIdList.size() == 0) {
//			return;
//		}
//
//		List<Mission> missionList = missionRepository.getMissionList();
//
//		List<DailyMissionBaseRes> scheduledList = new ArrayList<>();
//
//		Random random = new Random();
//		Set<Integer> pickedSet = new HashSet();
//
//		for (int i = 0; i < memberIdList.size(); i++) {
//			int memberId = memberIdList.get(i);
//			pickedSet.clear();
//			int cnt = 0;
//			while (cnt < 3) {
//				int index = random.nextInt(missionList.size());
//				if (pickedSet.contains(index)) {
//					// 다시 뽑아야 함
//					continue;
//				}
//				// 넣어도 됨
//				DailyMissionBaseRes dailyMission = DailyMissionBaseRes.builder()
//					.missionId(missionList.get(index).getMissionId())
//					.memberId(memberId)
//					.isFinish(false)
//					.build();
//				scheduledList.add(dailyMission);
//				pickedSet.add(index);
//				cnt++;
//			}
//		}
//
//		// template으로 교체
//		if (scheduledList.size() != 0) {
//
//			missionBulkRepository.saveAllDailyMission(scheduledList);
//		}

		// 알림 전송 kafka(notification에서는 알림테이블에 저장 + 실시간 알림 전송)

		NotificationEventDto eventDto = NotificationEventDto.builder()
			.eventType("DailyMissionSchedule")
			.memberId(-1)
			.build();

		ObjectMapper objectMapper = new ObjectMapper();
		String jsonString;
		// 객체를 JSON 문자열로 변환
		try {
			jsonString = objectMapper.writeValueAsString(eventDto);
			notificationEventDtoKafkaTemplate.send("notification-topic", jsonString);

		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

//		BaseResponse<Integer> res = notificationClient.dailyMissionScheduling();
		log.info("스케쥴링 완료 알림 전송. userId : 전체");


	}


	//	----------------------------- 일반 메서드 -------------------------------------
	private boolean isValidRequestTime() {
		LocalDateTime now = LocalDateTime.now();
		if (now.getHour() == 0 &&
			now.getMinute() >= 0 && now.getMinute() < 5) {
//			System.out.println("현재 시각은 12시00분00초와 12시05분00초 미만입니다.");
			return false;
		} else {
//			System.out.println("현재 시각은 12시00분00초와 12시05분00초 미만이 아닙니다.");
			return true;
		}
	}
}
