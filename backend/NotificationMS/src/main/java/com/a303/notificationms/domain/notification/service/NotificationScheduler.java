package com.a303.notificationms.domain.notification.service;

import com.a303.notificationms.domain.Emitter.repository.EmitterRepository;
import com.a303.notificationms.domain.notification.dto.response.CountNotificationMessageRes;
import com.a303.notificationms.domain.notification.dto.response.HeartBeatNotificationMessageRes;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationScheduler {

	private final EmitterRepository emitterRepository;

	@Scheduled(fixedRate = 180000) // 3분마다 heartbeat 메세지 전달.
	public void sendHeartbeat() {

		Map<Integer, SseEmitter> sseEmitters = emitterRepository.findAll();
		sseEmitters.forEach((key, emitter) -> {
			try {
				HeartBeatNotificationMessageRes messageRes = HeartBeatNotificationMessageRes.builder()
					.type("HEARTBEAT")
					.localDateTime(LocalDateTime.now())
					.build();
				emitter.send(SseEmitter.event().name("message").data(messageRes));
				log.info(key+"에게 하트비트 메세지 전송");
			} catch (IOException e) {
				emitterRepository.removeByMemberId(key);
				log.error("하트비트 전송 실패: {}", e.getMessage());
			}
		});
	}

}
