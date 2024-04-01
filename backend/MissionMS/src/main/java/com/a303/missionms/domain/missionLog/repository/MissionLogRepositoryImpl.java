package com.a303.missionms.domain.missionLog.repository;

import static com.a303.missionms.domain.mission.QMission.mission;
import static com.a303.missionms.domain.missionLog.QMissionLog.missionLog;
import static com.querydsl.core.types.dsl.Expressions.dateOperation;
import static com.querydsl.core.types.dsl.Expressions.dateTemplate;
import static java.util.Objects.requireNonNull;

import com.a303.missionms.domain.mission.dto.response.MissionBaseRes;
import com.a303.missionms.domain.mission.dto.response.MissionLogBaseRes;
import com.querydsl.core.types.Ops;
import com.querydsl.core.types.Ops.DateTimeOps;
import com.querydsl.core.types.Projections;
import com.querydsl.core.types.dsl.DateOperation;
import com.querydsl.core.types.dsl.NumberExpression;
import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.persistence.EntityManager;
import java.util.Date;
import java.util.List;
import org.springframework.stereotype.Repository;

@Repository
public class MissionLogRepositoryImpl implements MissionLogRepositoryCustom {

	private final JPAQueryFactory queryFactory;

	public MissionLogRepositoryImpl(EntityManager em) {
		this.queryFactory = new JPAQueryFactory(em);
	}


	@Override
	public List<MissionLogBaseRes> getMyReport(int memberId, int year, int month) {

		String yearMonthString = String.format("%d-%02d", year, month);
		DateOperation<Integer> monthExpression = dateOperation(Integer.class,
			Ops.DateTimeOps.MONTH, requireNonNull(missionLog.missionDate));
		DateOperation<Integer> yearExpression = dateOperation(Integer.class,
			DateTimeOps.YEAR, requireNonNull(missionLog.missionDate));

		return queryFactory.select(
				Projections.constructor(MissionLogBaseRes.class,
					missionLog.missionLogId,
					Projections.constructor(MissionBaseRes.class,
						mission.missionId,
						mission.name,
						mission.category
					),
					missionLog.memberId,
					missionLog.missionDate,
					missionLog.isFinish)
			)
			.from(missionLog)
			.join(mission).on(mission.missionId.eq(missionLog.mission.missionId))
			.where(missionLog.memberId.eq(memberId)
				.and(yearExpression.eq(year))
				.and(monthExpression.eq(month))
			)
			.fetch();
	}
}
