package com.a303.missionms.domain.mission.service;

import com.a303.missionms.domain.mission.dto.response.CategoryMissionRes;
import com.a303.missionms.domain.mission.dto.response.MissionListRes;
import com.a303.missionms.global.exception.BaseExceptionHandler;
import java.io.IOException;
import java.util.List;

public interface MissionService {

	MissionListRes getMissionList() throws BaseExceptionHandler, IOException;

	void welcome() throws BaseExceptionHandler;

}
