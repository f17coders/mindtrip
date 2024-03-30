package com.a303.notificationms.domain.notification.controller;

import com.a303.notificationms.domain.notification.service.NotificationService;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/notifications")
@RequiredArgsConstructor
@CrossOrigin("*")
@Slf4j
public class NotificationController {

	private final NotificationService notificationService;

	// 메시지 알림
	@GetMapping("/v0/subscribe")
	public SseEmitter subscribe(
		@RequestHeader("x-member-id") int memberId
	) {
		SseEmitter sseEmitter = notificationService.subscribe(memberId);

		notificationService.notifyCnt(memberId);

		return sseEmitter;
	}


}
