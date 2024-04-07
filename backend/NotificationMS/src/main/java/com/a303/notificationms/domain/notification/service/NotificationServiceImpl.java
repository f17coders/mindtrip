package com.a303.notificationms.domain.notification.service;

import com.a303.notificationms.domain.domain.Domain;
import com.a303.notificationms.domain.domain.repository.DomainRepository;
import com.a303.notificationms.domain.memberToken.MemberToken;
import com.a303.notificationms.domain.memberToken.repository.MemberTokenRepository;
import com.a303.notificationms.domain.notification.Notification;
import com.a303.notificationms.domain.notification.dto.NotificationEventDto;
import com.a303.notificationms.domain.notification.dto.request.FCMNotificationReq;
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
import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.messaging.FirebaseMessaging;
import com.google.firebase.messaging.FirebaseMessagingException;
import com.google.firebase.messaging.Message;
import jakarta.transaction.Transactional;
import java.io.IOException;
import java.io.InputStream;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.common.protocol.types.Field.Str;
import org.springframework.core.io.ClassPathResource;
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

	private final NotificationRepository notificationRepository;
	private final DomainRepository domainRepository;
	private final MemberClient memberClient;
	private final MemberTokenRepository memberTokenRepository;

	private final NotificationBulkRepository notificationBulkRepository;

	private final FirebaseMessaging firebaseMessaging;


	@Override
	public List<NotificationMessageRes> findNotificationsByMemberId(int memberId) {

//		Pageable pageable = PageRequest.of(0, 10);
//		List<Notification> notifications = notificationRepository.findByMemberId(memberId,
//			pageable);

		List<Notification> notifications = notificationRepository.findByMemberId(memberId);

		List<NotificationMessageRes> notificationMessageResList = new ArrayList<>();
		for (Notification noti : notifications
		) {
			NotificationMessageRes res = NotificationMessageRes.builder()
				.type("NOTIFICATION")
				.message(noti.getContent())
				.isWritten(noti.isWritten())
				.localDateTime(noti.getCreateTime())
				.build();
			notificationMessageResList.add(res);
		}

		return notificationMessageResList;
	}

	@Override
	public List<NotificationMessageRes> setIsWrittenTrue(int memberId) {

		List<NotificationMessageRes> notificationMessageResList = findNotificationsByMemberId(
			memberId);

		notificationBulkRepository.updateIsWrittenTrue(memberId);

		return notificationMessageResList;
	}

	//	--------------------- kafka listener ------------------------

	@KafkaListener(topics = "notification-topic", groupId = "group_1")
	public void consumeNotificationTopic(String message) {

		NotificationEventDto messageRes;
		try {
			ObjectMapper objectMapper = new ObjectMapper();
			messageRes = objectMapper.readValue(message, NotificationEventDto.class);
			log.error(messageRes.toString());
		} catch (JsonMappingException e) {
			// log
			throw new RuntimeException(e);
		} catch (JsonProcessingException e) {
			// log
			throw new RuntimeException(e);
		}

//		System.out.println(message);
		String content = "";
		String type = messageRes.eventType();
		if (type.equals("DailyMissionSchedule")) {
			dailyMissionScheduleEventHandler();
		}
		// 채팅방에 누가 입장했어요 : 방 주인에게 알림 ENTER
		else if (type.equals("Consult-ENTER")) {
			content = "고민상담소 채팅방에 상담자가 입장했어요.";
		}
		// 채팅방에 누가 나갔어요 : 방 주인에게 알림 EXIT
		else if (type.equals("Consult-EXIT")) {
			content = "상담자가 퇴장했어요. 다른 상담자를 얼른 찾아드릴게요.";
		}
		// 고민이 종료되었어요 : 참여자에게 알림 END
		else if (type.equals("Consult-END")) {
			content = "고민상담소 채팅이 종료되었어요. 친절한 상담 감사합니다.";
		}
		// 님 쫓겨났어요 : 참여자에게 알림 BANNED
		else if (type.equals("Consult-BANNED")) {
			content = "고민상담소 채팅방에서 퇴장당하셨습니다.";
		}
		// 카톡 왔어요 : 그냥 왔다는것만 알려주자. TALK
		else if (type.equals("Consult-TALK")) {
			// TODO 메시지 작성
		} else {
			throw new BaseExceptionHandler(type + "에 해당하는 type은 존재하지 않습니다.",
				ErrorCode.NOT_VALID_CODE);
		}

		makeConsultNotification(content, messageRes.memberId());


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

		for (int memberId : memberIdList
		) {
			// 유저의 모든 토큰을 가져온다.
			MemberToken memberToken = memberTokenRepository.findByMemberId(memberId);
			if (memberToken == null) return;

			Long notificationCnt = notificationRepository.countByMemberIdAndAndIsWritten(memberId,
					false);

			// 발송할 토큰이 있는 경우에만 전송
			Iterator<String> memberTokenIterator = memberToken.getTokens().iterator();
			while (memberTokenIterator.hasNext()) {
				String token = memberTokenIterator.next();
				Message message1 = Message.builder()
						.setToken(token)
						.putData("type", "NOTIFICATION")
						.putData("message", now + " 날짜의 미션이 추가되었습니다. :)")
						.putData("isWritten", String.valueOf(false))
						.putData("localDateTime", LocalDateTime.now().toString())
						.build();

				Message message2 = Message.builder()
						.setToken(token)
						.putData("type", "COUNT")
						.putData("count", notificationCnt.toString())
						.putData("localDateTime", LocalDateTime.now().toString())
						.build();

				try {
					firebaseMessaging.send(message1);
					firebaseMessaging.send(message2);
				} catch (FirebaseMessagingException e) {

					memberToken.getTokens().remove(token);
//					throw new BaseExceptionHandler(e.toString(), ErrorCode.FCM_IO_EXCEPTION);
				}

			}

		}
	}

	@Override
	public void makeConsultNotification(String content, int memberId) {

		Domain domain = domainRepository.findByName("고민상담소");
		Notification notification = Notification.createNotification(
			memberId,
			domain,
			false,
			content
		);
		notificationRepository.save(notification);


		// 유저의 모든 토큰을 가져온다.
		MemberToken memberToken = memberTokenRepository.findByMemberId(memberId);
		if (memberToken == null) return;

		Long notificationCnt = notificationRepository.countByMemberIdAndAndIsWritten(memberId,
				false);

		// 발송할 토큰이 있는 경우에만 전송
		Iterator<String> memberTokenIterator = memberToken.getTokens().iterator();
		while (memberTokenIterator.hasNext()) {
			String token = memberTokenIterator.next();
			Message message1 = Message.builder()
					.setToken(token)
					.putData("type", "NOTIFICATION")
					.putData("message", notification.getContent())
					.putData("isWritten", String.valueOf(notification.isWritten()))
					.putData("localDateTime", LocalDateTime.now().toString())
					.build();

			Message message2 = Message.builder()
					.setToken(token)
					.putData("type", "COUNT")
					.putData("count", notificationCnt.toString())
					.putData("localDateTime", LocalDateTime.now().toString())
					.build();

			try {
				firebaseMessaging.send(message1);
				firebaseMessaging.send(message2);
			} catch (FirebaseMessagingException e) {

				memberToken.getTokens().remove(token);
//					throw new BaseExceptionHandler(e.toString(), ErrorCode.FCM_IO_EXCEPTION);
			}

		}



	}


	@Override
	public void makeNotification(int memberId) {

		// Notification 저장
		Domain domain = domainRepository.findByName("임시");
		LocalDate now = LocalDate.now();

		Notification notification = Notification.createNotification(
			memberId,
			domain,
			false,
			now + " 임시 알림 발송!"
		);

		notificationRepository.save(notification);

		// 유저의 모든 토큰을 가져온다.
		MemberToken memberToken = memberTokenRepository.findByMemberId(memberId);
		if (memberToken == null) return;

		Long notificationCnt = notificationRepository.countByMemberIdAndAndIsWritten(memberId,
				false);

		// 발송할 토큰이 있는 경우에만 전송
		Iterator<String> memberTokenIterator = memberToken.getTokens().iterator();
		while (memberTokenIterator.hasNext()) {
			String token = memberTokenIterator.next();
			Message message1 = Message.builder()
					.setToken(token)
					.putData("type", "NOTIFICATION")
					.putData("message", notification.getContent())
					.putData("isWritten", String.valueOf(notification.isWritten()))
					.putData("localDateTime", LocalDateTime.now().toString())
					.build();

			Message message2 = Message.builder()
					.setToken(token)
					.putData("type", "COUNT")
					.putData("count", notificationCnt.toString())
					.putData("localDateTime", LocalDateTime.now().toString())
					.build();

			try {
				firebaseMessaging.send(message1);
				firebaseMessaging.send(message2);
			} catch (FirebaseMessagingException e) {

				memberToken.getTokens().remove(token);
//					throw new BaseExceptionHandler(e.toString(), ErrorCode.FCM_IO_EXCEPTION);
			}

		}

	}

	@Override
	public void sendCountNotification(int memberId) {

		// 유저의 모든 토큰을 가져온다.
		MemberToken memberToken = memberTokenRepository.findByMemberId(memberId);
		if (memberToken == null) return;

		// 발송할 토큰이 있는 경우에만 전송
		// memberId로 저장된 미개봉 notification 개수 조회
		Long notificationCnt = notificationRepository.countByMemberIdAndAndIsWritten(memberId,
				false);

		Iterator<String> memberTokenIterator = memberToken.getTokens().iterator();
		while (memberTokenIterator.hasNext()) {
			String token = memberTokenIterator.next();
			Message message = Message.builder()
					.setToken(token)
					.putData("type", "COUNT")
					.putData("count", notificationCnt.toString())
					.putData("localDateTime", LocalDateTime.now().toString())
					.build();

			try {
				firebaseMessaging.send(message);
			} catch (FirebaseMessagingException e) {
				memberToken.getTokens().remove(token);
//				throw new BaseExceptionHandler(e.toString(), ErrorCode.FCM_IO_EXCEPTION);
			}

		}

	}

	@Override
	public void saveToken(int memberId, String token) {

//		널 -> 새로 넣어 / 널아님, 있음 -> 종료 / 널아님, 없음 -> 추가한다.

		// 멤버id로 일단 가져와
		MemberToken memberToken = memberTokenRepository.findByMemberId(memberId);
		if (memberToken != null) {
			Set<String> memberTokenSet = memberToken.getTokens();
			if(!memberTokenSet.contains(token)) {
				// 없음 -> 추가한다.
				memberTokenSet.add(token);
				memberToken.setTokens(memberTokenSet);
				memberTokenRepository.save(memberToken);
			}
			// 이미 존재하는 토큰 -> 에러 반환
			else {
				throw new BaseExceptionHandler(ErrorCode.GOOGLE_TOKEN_ALREADY_EXISTS);
			}
		}
		else {
			// 아예 새로 MemberToken 만들어야함
			Set<String> tokens = new HashSet<>();
			tokens.add(token);
			MemberToken tempMemberToken = MemberToken.createMemberToken(memberId, tokens);
			memberTokenRepository.save(tempMemberToken);
		}

	}

}
