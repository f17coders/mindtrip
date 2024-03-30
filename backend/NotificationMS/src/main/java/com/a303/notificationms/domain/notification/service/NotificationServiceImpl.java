package com.a303.notificationms.domain.notification.service;

import com.a303.notificationms.domain.Emitter.repository.EmitterRepository;
import com.a303.notificationms.domain.domain.Domain;
import com.a303.notificationms.domain.domain.repository.DomainRepository;
import com.a303.notificationms.domain.notification.Notification;
import com.a303.notificationms.domain.notification.dto.NotificationEventDto;
import com.a303.notificationms.domain.notification.dto.response.CountNotificationMessageRes;
import com.a303.notificationms.domain.notification.dto.response.NotificationMessageRes;
import com.a303.notificationms.domain.notification.repository.NotificationBulkRepository;
import com.a303.notificationms.domain.notification.repository.NotificationRepository;
import com.a303.notificationms.global.client.MemberClient;
import com.a303.notificationms.global.exception.BaseExceptionHandler;
import com.a303.notificationms.global.exception.code.ErrorCode;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import java.io.IOException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class NotificationServiceImpl implements NotificationService {

//	@Autowired
//	private final EntityManager em;

	private final EmitterRepository emitterRepository;
	private final NotificationRepository notificationRepository;
	private final DomainRepository domainRepository;
	private final MemberClient memberClient;

	private final NotificationBulkRepository notificationBulkRepository;


	// 메시지 알림
	public SseEmitter subscribe(int memberId) {

		// 1. 현재 클라이언트를 위한 sseEmitter 객체 생성
		SseEmitter sseEmitter = new SseEmitter((long)(60 * 1000));

		// 2. 연결
		try {
			sseEmitter.send(SseEmitter.event().name("connect"));
			log.info("connect");
		} catch (IOException e) {
			e.printStackTrace();
		}

		// 3. 저장
		emitterRepository.save(memberId, sseEmitter);

		// 4. 연결 종료 처리
		sseEmitter.onCompletion(
			() -> emitterRepository.removeByMemberId(memberId));    // sseEmitter 연결이 완료될 경우
		sseEmitter.onTimeout(() -> emitterRepository.removeByMemberId(
			memberId));        // sseEmitter 연결에 타임아웃이 발생할 경우
		sseEmitter.onError((e) -> emitterRepository.removeByMemberId(
			memberId));        // sseEmitter 연결에 오류가 발생할 경우

		return sseEmitter;
	}

	public void notifyCnt(int memberId) {

		// memberId로 저장된 미개봉 notification 개수 조회
		Long notificationCnt = notificationRepository.countByMemberIdAndAndIsWritten(memberId,
			false);

		// 7. Map 에서 userId 로 사용자 검색
		SseEmitter sseEmitterReceiver = emitterRepository.findByMemberId(memberId);
		if (sseEmitterReceiver == null) {
			throw new BaseExceptionHandler("SseEmitter 발견 불가", ErrorCode.NOT_FOUND_ERROR);
		}

		// 8. 알림 메시지 전송 및 해체
		try {
			CountNotificationMessageRes messageRes = CountNotificationMessageRes.builder()
				.type("COUNT")
				.count(notificationCnt)
				.build();
			sseEmitterReceiver.send(SseEmitter.event().name("message").data(messageRes));

		} catch (Exception e) {
			emitterRepository.removeByMemberId(memberId);
		}
	}

	@Override
	public List<NotificationMessageRes> findNotificationsByMemberId(int memberId) {

		Pageable pageable = PageRequest.of(0, 5);
		List<Notification> notifications = notificationRepository.findByMemberId(memberId, pageable);

		List<NotificationMessageRes> notificationMessageResList = new ArrayList<>();
		for (Notification noti:notifications
		) {
			NotificationMessageRes res = NotificationMessageRes.builder()
					.type("NOTIFICATION")
					.message(noti.getContent())
					.build();
			notificationMessageResList.add(res);
		}

		return notificationMessageResList;
	}

	@Override
	public void setIsWrittenTrue(int memberId) {
		notificationBulkRepository.updateIsWrittenTrue(memberId);

	}

	//	--------------------- kafka listener ------------------------

	@KafkaListener(topics = "notification-topic", groupId = "group_1")
	public void consumeNotificationTopic(String message) {

		NotificationEventDto messageRes;
		try {
			ObjectMapper objectMapper = new ObjectMapper();
			messageRes = objectMapper.readValue(message, NotificationEventDto.class);
			System.out.println(messageRes);
		} catch (JsonMappingException e) {
			// log
			throw new RuntimeException(e);
		} catch (JsonProcessingException e) {
			// log
			throw new RuntimeException(e);
		}

//		System.out.println(message);

		String type = messageRes.eventType();
		if (type.equals("DailyMissionSchedule")) {
			dailyMissionScheduleEventHandler();
		}

	}

	@Override
	public void dailyMissionScheduleEventHandler() {
		// Notification 저장
		Domain domain = domainRepository.findByName("공지");
		LocalDate now = LocalDate.now();

		List<Integer> memberIdList = memberClient.getMemberIdList().getResult();
		if (memberIdList.size() == 0) {
			return;
		}

		List<Notification> notifications = new ArrayList<>();
		for (int memberId : memberIdList) {
			notifications.add(Notification.createNotification(
					memberId,
					domain,
					false,
					now + " 날짜의 미션이 추가되었습니다. :)"
				)
			);
		}
		notificationRepository.saveAll(notifications);

		// 실시간 알람
		Map<Integer, SseEmitter> sseEmitters = emitterRepository.findAll();
		for (Entry<Integer, SseEmitter> entry : sseEmitters.entrySet()) {
			try {
				SseEmitter sseEmitterReceiver = entry.getValue();
				NotificationMessageRes messageRes = NotificationMessageRes.builder()
					.type("NOTIFICATION")
					.message(now + " 날짜의 미션이 추가되었습니다. :)")
					.build();
				sseEmitterReceiver.send(SseEmitter.event().name("message").data(messageRes));

			} catch (Exception e) {
				emitterRepository.removeByMemberId(entry.getKey());
			}
		}
	}

}
