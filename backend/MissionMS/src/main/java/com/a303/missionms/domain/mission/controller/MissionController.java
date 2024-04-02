package com.a303.missionms.domain.mission.controller;

import com.a303.missionms.domain.dailyMission.service.DailyMissionService;
import com.a303.missionms.domain.mission.dto.request.MyTableMissionDTO;
import com.a303.missionms.domain.mission.dto.response.MissionListRes;
import com.a303.missionms.domain.mission.dto.response.MissionReportRes;
import com.a303.missionms.domain.mission.dto.response.MyTableMissionRes;
import com.a303.missionms.domain.mission.service.MissionService;
import com.a303.missionms.domain.missionLog.service.MissionLogService;
import com.a303.missionms.global.api.response.BaseResponse;
import com.a303.missionms.global.exception.code.SuccessCode;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/missions")
@RequiredArgsConstructor
@CrossOrigin("*")
@Slf4j
public class MissionController {

	private final MissionService missionService;
	private final DailyMissionService dailyMissionService;
	private final MissionLogService missionLogService;

	//    @Operation(summary = "미션리스트조회")
	@GetMapping("/v0")
	public ResponseEntity<BaseResponse<MissionListRes>> getMissionList() throws IOException {

		log.debug("missions/v0 GET api accepted");

		MissionListRes missionListRes = missionService.getMissionList();

		log.debug("missions/v0 GET api succeed");

		return BaseResponse.success(SuccessCode.SELECT_SUCCESS, missionListRes);
	}

	//    @Operation(summary = "마이테이블관리")
	@PutMapping("/v1/mytable")
	public ResponseEntity<BaseResponse<List<MyTableMissionRes>>> putMyTableMissions(
		@RequestHeader("x-member-id") int memberId,
		@RequestBody HashMap<Integer, MyTableMissionDTO> myTableMissionDTOMap) throws IOException {

		log.debug("missions/v1/mytable PUT api accepted with memberId:{}", memberId);

//		for (Entry<Integer, MyTableMissionDTO> m : myTableMissionDTOMap.entrySet()) {
//			System.out.println(m);
//		}

		List<MyTableMissionRes> myTableMissionDTOList = dailyMissionService.putMyTableMissions(
			memberId, myTableMissionDTOMap);

		log.debug("missions/v1/mytable PUT api succeed with memberId:{}", memberId);

		return BaseResponse.success(SuccessCode.INSERT_SUCCESS, myTableMissionDTOList);
	}

	//    @Operation(summary = "마이테이블조회")
	@GetMapping("/v1/mytable")
	public ResponseEntity<BaseResponse<List<MyTableMissionRes>>> getMyTableMissions(
		@RequestHeader("x-member-id") int memberId) throws IOException {

		log.debug("missions/v1/mytable GET api accepted with memberId:{}", memberId);

		List<MyTableMissionRes> myTableMissionDTOList = dailyMissionService.getMyTableMissions(
			memberId);

		log.debug("missions/v1/mytable GET api succeed with memberId:{}", memberId);

		return BaseResponse.success(SuccessCode.SELECT_SUCCESS, myTableMissionDTOList);
	}

	//    @Operation(summary = "미션완료변경") TODO 최적화 완료, redis 필요
	@PostMapping("/v1/mytable/{missionId}")
	public ResponseEntity<BaseResponse<Integer>> completeMission(
		@RequestHeader("x-member-id") int memberId,
		@PathVariable("missionId") int missionId) throws IOException {

		log.debug("missions/v1/mytable/{missionId} POST api accepted with memberId:{} missionId:{}",
			memberId, missionId);

		int dailyMissionId = dailyMissionService.completeMission(memberId, missionId);

		log.debug("missions/v1/mytable/{missionId} POST api succeed with memberId:{} missionId:{}",
			memberId, missionId);

		return BaseResponse.success(SuccessCode.UPDATE_SUCCESS, dailyMissionId);
	}

	@GetMapping("/v0/schedule")
	public ResponseEntity<BaseResponse<Integer>> completeMission() throws IOException {

		dailyMissionService.dailyMissionRecommend();

		return BaseResponse.success(SuccessCode.UPDATE_SUCCESS, 1);
	}

	@GetMapping("/v1/report")
	public ResponseEntity<BaseResponse<MissionReportRes>> getMyMonthlyReport(
		@RequestHeader("x-member-id") int memberId,
		@RequestParam("year") int year,
		@RequestParam("month") int month
	) throws IOException {

		log.debug("missions/v1/report GET api accepted with memberId:{}", memberId);

		MissionReportRes missionReportRes = missionLogService.getMissionReport(memberId, year,
			month);

		log.debug("missions/v1/report GET api succeed with memberId:{}", memberId);

		return BaseResponse.success(SuccessCode.SELECT_SUCCESS, missionReportRes);
	}

//	@GetMapping("/v1/count")
//	public ResponseEntity<BaseResponse<Long>> getMyMonthlyReport(
//		@RequestHeader("x-member-id") int memberId
//	) throws IOException {
//
//		long cnt = missionLogService.getCompletedMissionCount(memberId);
//
//		return BaseResponse.success(SuccessCode.SELECT_SUCCESS, cnt);
//	}

	//	---------------------------------------- Method -------------------------------------------


}
