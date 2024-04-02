package com.a303.missionms.domain.mission.repository;

import com.a303.missionms.domain.mission.Mission;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface MissionRepository extends JpaRepository<Mission, Integer> {

	List<Mission> findAll();

//	@EntityGraph(attributePaths = {"missionIdList"})
//	@Query("SELECT DISTINCT m FROM Mission m WHERE m.missionId IN :missionIdList")
//	List<Mission> getMissionsByMissionIdIn(
//		@Param("missionIdList") List<Integer> missionIdList);

	Mission findMissionByMissionId(int missionId);

	@Query("SELECT DISTINCT m FROM Mission m")
	List<Mission> getMissionList();

}
