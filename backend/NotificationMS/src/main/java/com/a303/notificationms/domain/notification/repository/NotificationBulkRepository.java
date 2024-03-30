package com.a303.notificationms.domain.notification.repository;


import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.time.format.DateTimeFormatter;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementSetter;
import org.springframework.stereotype.Repository;

@Repository
@RequiredArgsConstructor
@Slf4j
public class NotificationBulkRepository {

	private final JdbcTemplate jdbcTemplate;

	public void updateIsWrittenTrue(int memberId) {
		String sql = "update notification set is_written=true where memberId = ? and is_written=false";

		jdbcTemplate.update(sql, new PreparedStatementSetter() {
			@Override
			public void setValues(PreparedStatement ps) throws SQLException {
				ps.setInt(1, memberId);
			}
		});
	}


}
