package com.a303.notificationms.domain.notification.controller;

import com.a303.notificationms.domain.notification.dto.response.NotificationMessageRes;
import com.a303.notificationms.domain.notification.service.NotificationService;
import com.a303.notificationms.global.api.response.BaseResponse;
import com.a303.notificationms.global.exception.code.SuccessCode;
import java.time.Duration;
import java.time.LocalTime;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/notifications")
@RequiredArgsConstructor
@CrossOrigin("*")
@Slf4j
public class NotificationController {

	private final NotificationService notificationService;

	// 메시지 알림
	@GetMapping("/v1/subscribe")
	public SseEmitter subscribe(
		@RequestHeader("x-member-id") int memberId
	) {
		SseEmitter sseEmitter = notificationService.subscribe(memberId);

		notificationService.notifyCnt(memberId);

		return sseEmitter;
	}

	// 알림 읽음 처리
	@PostMapping("/v1")
	public ResponseEntity<BaseResponse<Integer>> setIsWrittenTrue(
			@RequestHeader("x-member-id") int memberId
	) {

		notificationService.setIsWrittenTrue(memberId);


		return BaseResponse.success(SuccessCode.UPDATE_SUCCESS, 1);
	}

	@GetMapping("/v1")
	public ResponseEntity<BaseResponse<List<NotificationMessageRes>>> read(
			@RequestHeader("x-member-id") int memberId
	) {

		List<NotificationMessageRes> notifications = notificationService.findNotificationsByMemberId(memberId);


		return BaseResponse.success(SuccessCode.SELECT_SUCCESS, notifications);
	}

//	-------------------- feign -----------------------

	@PostMapping("/v0/dailymission")
	public ResponseEntity<BaseResponse<Integer>> dailyMissionScheduling(
	) {

		notificationService.dailyMissionScheduleEventHandler();

		return BaseResponse.success(SuccessCode.INSERT_SUCCESS, 1);
	}


//	@GetMapping("/v1/subscribe")
//	public ResponseEntity<BaseResponse<String>> subscribe(
//			@RequestHeader("x-member-id") int memberId
//	) {
//
//		return BaseResponse.success(SuccessCode.INSERT_SUCCESS, "im good!");
//	}


}
