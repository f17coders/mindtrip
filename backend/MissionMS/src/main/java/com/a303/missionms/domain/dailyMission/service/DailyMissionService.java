package com.a303.missionms.domain.dailyMission.service;

import com.a303.missionms.domain.mission.dto.request.MyTableMissionDTO;
import com.a303.missionms.domain.mission.dto.response.MissionReportRes;
import com.a303.missionms.domain.mission.dto.response.MyTableMissionRes;
import com.a303.missionms.global.exception.BaseExceptionHandler;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public interface DailyMissionService {

	List<MyTableMissionRes> putMyTableMissions(int memberId,
		HashMap<Integer, MyTableMissionDTO> myTableMissionDTOMap)
		throws BaseExceptionHandler, IOException;

	List<MyTableMissionRes> getMyTableMissions(int memberId)
		throws BaseExceptionHandler, IOException;

	int completeMission(int memberId, int missionId) throws BaseExceptionHandler, IOException;

	void dailyMissionRecommend() throws BaseExceptionHandler;

}
