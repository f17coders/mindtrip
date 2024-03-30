package com.a303.notificationms.domain.notification.repository;

import com.a303.notificationms.domain.notification.Notification;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface NotificationRepository extends JpaRepository<Notification, Integer> {

	@Query("select n from Notification n where n.memberId=:memberId order by n.notificationId desc")
	List<Notification> findByMemberId(@Param("memberId") int memberId, Pageable pageable);

	Long countByMemberIdAndAndIsWritten(int memberId, boolean isWritten);



}
