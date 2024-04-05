package com.a303.notificationms.domain.notification.service;

import com.a303.notificationms.domain.notification.dto.request.FCMNotificationReq;
import com.a303.notificationms.domain.notification.dto.response.NotificationMessageRes;
import java.util.List;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

public interface NotificationService {

	SseEmitter subscribe(int memberId);

	void notifyCnt(int memberId);

	List<NotificationMessageRes> findNotificationsByMemberId(int memberId);

	List<NotificationMessageRes> setIsWrittenTrue(int memberId);

	void dailyMissionScheduleEventHandler();

	void makeConsultNotification(String type, int memberId);

	void makeNotification(int memberId);

	String sendNotificationByToken(FCMNotificationReq reqDto);





	}
