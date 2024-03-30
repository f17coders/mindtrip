package com.a303.notificationms.domain.notification.controller;

import com.a303.notificationms.domain.notification.service.NotificationService;
import com.a303.notificationms.global.api.response.BaseResponse;
import com.a303.notificationms.global.exception.code.SuccessCode;
import java.time.Duration;
import java.time.LocalTime;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
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

	@GetMapping(value = "/v0/stream-sse", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
	public Flux<ServerSentEvent<String>> streamSseMvc() {
		return Flux.interval(Duration.ofSeconds(1))
				.map(sequence -> {
							if (sequence > 50) {
								return ServerSentEvent.<String>builder().comment("stop sending").build();
							} else {
								// 正常发送 SSE 事件
								return ServerSentEvent.<String>builder()
										.id(String.valueOf(sequence))
										.event("message")
										.data("SSE - " + LocalTime.now())
										.build();
							}
						}
				)
				.doOnCancel(() -> System.out.println("SSE connection cancelled"))
				.doOnError(Throwable::printStackTrace)
				.doOnComplete(() -> System.out.println("SSE connection completed"));
	}


//	@GetMapping("/v1/subscribe")
//	public ResponseEntity<BaseResponse<String>> subscribe(
//			@RequestHeader("x-member-id") int memberId
//	) {
//
//		return BaseResponse.success(SuccessCode.INSERT_SUCCESS, "im good!");
//	}


}
