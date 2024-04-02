package com.a303.consultms.domain.consult.service;

import static com.a303.consultms.global.exception.code.ErrorCode.ALREADY_CLOSED_EXCEPTION;
import static com.a303.consultms.global.exception.code.ErrorCode.ALREADY_CONSULT_LIKE_EXISTS;
import static com.a303.consultms.global.exception.code.ErrorCode.ALREADY_FULL_CONSULTROOM;
import static com.a303.consultms.global.exception.code.ErrorCode.CHANNEL_NOT_FOUND_EXCEPTION;
import static com.a303.consultms.global.exception.code.ErrorCode.NOT_FOUND_CHANNEL_EXCEPTION;
import static com.a303.consultms.global.exception.code.ErrorCode.NOT_FOUND_CONSULT_EXCEPTION;
import static com.a303.consultms.global.exception.code.ErrorCode.NOT_FOUND_CONSULT_LIKE_EXISTS;
import static com.a303.consultms.global.exception.code.ErrorCode.UNAUTHORIZED_USER_EXCEPTION;

import com.a303.consultms.domain.channel.Channel;
import com.a303.consultms.domain.channel.repository.ChannelRepository;
import com.a303.consultms.domain.consult.Consult;
import com.a303.consultms.domain.consult.ConsultCategory;
import com.a303.consultms.domain.consult.dto.request.ConsultCloseReq;
import com.a303.consultms.domain.consult.dto.request.ConsultRegisterReq;
import com.a303.consultms.domain.consult.dto.response.ConsultCategoryListRes;
import com.a303.consultms.domain.consult.dto.response.ConsultCategoryRes;
import com.a303.consultms.domain.consult.dto.response.ConsultChattingListRes;
import com.a303.consultms.domain.consult.dto.response.ConsultChattingRes;
import com.a303.consultms.domain.consult.dto.response.ConsultDetailRes;
import com.a303.consultms.domain.consult.dto.response.ConsultListRes;
import com.a303.consultms.domain.consult.repository.ConsultCategoryRepository;
import com.a303.consultms.domain.consult.repository.ConsultRepository;
import com.a303.consultms.domain.consultLike.LikeConsult;
import com.a303.consultms.domain.consultLike.repository.LikeConsultRepository;
import com.a303.consultms.domain.member.dto.response.MemberBaseRes;
import com.a303.consultms.domain.message.Message;
import com.a303.consultms.domain.notification.dto.response.NotificationEventDto;
import com.a303.consultms.global.api.response.BaseResponse;
import com.a303.consultms.global.client.MemberClient;
import com.a303.consultms.global.client.NotificationClient;
import com.a303.consultms.global.exception.BaseExceptionHandler;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.data.redis.core.HashOperations;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Log4j2
@Service
@RequiredArgsConstructor
public class ConsultServiceImpl implements ConsultService {

	private final ConsultRepository consultRepository;
	private final ConsultCategoryRepository consultCategoryRepository;
	private final MemberClient memberClient;
	private final ChannelRepository channelRepository;
	private final LikeConsultRepository likeConsultRepository;
	private final RedisTemplate<String, String> redisTemplate;
	private final NotificationClient notificationClient;

	private final KafkaTemplate<String, String> notificationEventDtoKafkaTemplate;

	//고민상담소 전체 조회
	@Override
	public ConsultListRes getConsultingRooms(int memberId) throws BaseExceptionHandler {
		// createtime 기준으로 내림차순 정렬
		List<Consult> consultList = consultRepository.findAllByOrderByCreateTimeDesc();

		if (consultList == null) {
			return null;
		}

		// Consult 객체의 리스트를 ConsultDetailRes 객체의 리스트로 변환
		List<ConsultDetailRes> consultDetailResList = consultList.stream().map(
			consult -> ConsultDetailRes.builder()
				.consultId(consult.getConsultId())
				.memberId(consult.getMemberId())
				.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
				.title(consult.getTitle())
				.content(consult.getContent())
				.categoryId(consult.getCategoryId())
				.isClosed(consult.isClosed())
				.isShared(consult.isShared())
				.canLike(isLike(consult.getConsultId(), memberId))
				.channelId(consult.getChannelId())
				.build()).collect(Collectors.toList());

		return ConsultListRes.builder().consultList(consultDetailResList)
			.build();
	}

	//입장가능한 고민상담소 리스트 조회
	@Override
	public ConsultListRes getAvailableConsultingRooms(int memberId) {
		// createtime 기준으로 내림차순 정렬
		List<Consult> consultList = consultRepository.findAllByChannelIdOrderByCreateTimeDesc(null);

		// Consult 객체의 리스트를 ConsultDetailRes 객체의 리스트로 변환
		List<ConsultDetailRes> consultDetailResList = consultList.stream().map(
			consult -> ConsultDetailRes.builder()
				.consultId(consult.getConsultId())
				.memberId(consult.getMemberId())
				.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
				.title(consult.getTitle())
				.content(consult.getContent())
				.categoryId(consult.getCategoryId())
				.isClosed(consult.isClosed())
				.isShared(consult.isShared())
				.canLike(isLike(consult.getConsultId(), memberId))
				.channelId(consult.getChannelId()).build()).collect(Collectors.toList());

		return ConsultListRes.builder().consultList(consultDetailResList)
			.build();
	}

	//고민상담소 등록
	@Override
	@Transactional
	public int registerConsultingRoom(ConsultRegisterReq consultRegisterReq, int memberId)
		throws BaseExceptionHandler {
		log.info("[ConsultService]: memberId = {}", memberId);

		//memberId로 memberClient에서 memberRes에 정보 담아 받아오기
		BaseResponse<MemberBaseRes> memberRes = memberClient.getMember(memberId);

		MemberBaseRes member = memberRes.getResult();

		String nickname = member.nickname();
		String title = consultRegisterReq.title();
		String content = consultRegisterReq.content();
		int categoryId = consultRegisterReq.categoryId();

		//고민상담소 등록
		Consult consult = Consult.createConsult(memberId, nickname, title, content, categoryId);
		consultRepository.save(consult);

		return consult.getConsultId();
	}

	//고민상담소에 채널 등록
	@Override
	@Transactional
//    public String registerChannel(ChannelReq channelReq, int consultId, int senderId) {
	public Channel registerChannel(int consultId, int senderId) throws BaseExceptionHandler {
		//senderId는 입장하는 사람 ID
		//고민상담소에 기존 채널 등록되어 있는지 확인
		Consult consult = consultRepository.findById(consultId).get();

		//고민상담소가 닫혀 있는지 확인 -> 닫혀있으면 에러 처리
		if (consult.isClosed()) {
			throw new BaseExceptionHandler(ALREADY_CLOSED_EXCEPTION);
		}

		//고민상담소 만든 사람이 자기 자신인 경우 에러 처리
		if (consult.getMemberId() == senderId) {
			throw new BaseExceptionHandler(UNAUTHORIZED_USER_EXCEPTION);
		}

		if (consult.getChannelId() != null) {
			//등록된 채널 있으면 예외 발생

			throw new BaseExceptionHandler(ALREADY_FULL_CONSULTROOM);

		}

		//등록된 채널 없으면 새로운 채널 생성하여 등록
		Map<String, String> receiver = new HashMap<>();
		//receiver 정보 받아오기
		MemberBaseRes receiverMember = memberClient.getMember(consult.getMemberId())
			.getResult();
		receiver.put("memberId", String.valueOf(consult.getMemberId()));
		receiver.put("nickname", receiverMember.nickname());

		Map<String, String> sender = new HashMap<>();
		//sender 정보 받아오기
		MemberBaseRes senderMember = memberClient.getMember(senderId).getResult();
		sender.put("memberId", String.valueOf(senderId));
		sender.put("nickname", senderMember.nickname());

		//새로운 채널 생성
		Channel channel = Channel.createChannel(consultId, receiver, sender, new ArrayList<>());
		//채널 저장
		channelRepository.save(channel);

		//생성된 채널의 ID를 consult에 저장
		consult.setChannelId(channel.getChannelId());
		consultRepository.save(consult);

		return channel;


	}

	//고민상담소 개별 조회
	@Override
	public ConsultDetailRes getConsultingRoom(String channelId) {

		Consult consult = consultRepository.findAllByChannelIdOrderByCreateTimeDesc(channelId)
			.get(0);

		return ConsultDetailRes.builder().consultId(consult.getConsultId())
			.memberId(consult.getMemberId())
			.nickname(consult.getNickname()).title(consult.getTitle()).content(consult.getContent())
			.categoryId(consult.getCategoryId()).isClosed(consult.isClosed())
			.isShared(consult.isShared()).canLike(consult.isCanLike())
			.channelId(channelId).build();
	}

	//고민상담소 종료
	@Override
	@Transactional
	public int closeConsultingRoom(int consultId, ConsultCloseReq consultCloseReq, int memberId)
		throws BaseExceptionHandler {

		Consult consult = consultRepository.findById(consultId)
			.orElseThrow(() -> new BaseExceptionHandler(NOT_FOUND_CONSULT_EXCEPTION));
		boolean isShared = consultCloseReq.isShared();
		log.info("[ConsultService]: isShared = {}", isShared);

		//상담소를 종료하려는 사람이 권한이 없는 경우 에러 처리
		if (consult.getMemberId() != memberId) {
			throw new BaseExceptionHandler(UNAUTHORIZED_USER_EXCEPTION);
		}

		//고민상담고가 이미 닫힌 경우
		if (consult.isClosed() == true) {
			throw new BaseExceptionHandler(ALREADY_CLOSED_EXCEPTION);
		}

		consult.setClosed(true); //종료 상태로 저장
		consult.setShared(isShared); //공유여부 저장
		consultRepository.save(consult);

		//해당 고민상담소에 현재 채널에도 공유여부 저장하기
		String channelId = consult.getChannelId();
		Channel channel = channelRepository.findById(channelId)
			.orElseThrow(() -> new BaseExceptionHandler(NOT_FOUND_CHANNEL_EXCEPTION));
		channel.setShared(isShared); //현재 채널의 공유여부 저장
		channel.setClosed(true); //채널 종료여부 저장
		channelRepository.save(channel);

		// TODO 알림 발생 : notificationms에 전송
//        notificationClient.consultNotification("END",
//            Integer.parseInt(channel.getReceiver().get("memberId")));

		return consult.getConsultId();
	}

	//고민상담소 카테고리 조회
	@Override
	public ConsultCategoryListRes getConsultCategoryList() {
		List<ConsultCategory> consultCategoryList = consultCategoryRepository.findAll();

		// ConsultCategory 객체의 리스트를 ConsultCategoryRes 객체의 리스트로 변환
		List<ConsultCategoryRes> consultCategoryRes = consultCategoryList.stream().map(
			category -> ConsultCategoryRes.builder().categoryId(category.getCategoryId())
				.categoryName(category.getCategoryName()).build()).collect(Collectors.toList());

		return ConsultCategoryListRes.builder().consultCategoryList(consultCategoryRes).build();
	}

	//공유된 고민 리스트 조회
	@Override
	public ConsultListRes getSharedConsultingRooms(int memberId) throws BaseExceptionHandler {
		List<Consult> consultList = consultRepository.findAllByIsSharedOrderByCreateTimeDesc(true);

		// Consult 객체의 리스트를 ConsultDetailRes 객체의 리스트로 변환
		List<ConsultDetailRes> consultDetailResList = consultList.stream().map(
			consult -> ConsultDetailRes.builder().consultId(consult.getConsultId())
				.memberId(consult.getMemberId())
				.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
				.title(consult.getTitle())
				.content(consult.getContent())
				.categoryId(consult.getCategoryId())
				.isClosed(consult.isClosed())
				.isShared(consult.isShared())
				.canLike(isLike(consult.getConsultId(), memberId))
				.channelId(consult.getChannelId())
				.build()).collect(Collectors.toList());

		// TODO: 전체 페이지 수를 계산하는 로직 추가 필요

		return ConsultListRes.builder().consultList(consultDetailResList)
//            .totalPages(totalPages)
			.build();
	}

	//고민상담소 나가기(퇴장)
	@Override
	public void exitConsultingRoom(int consultId, int sender) throws BaseExceptionHandler {

		//consultId 기반으로 현재 Consult 조회
		Consult consult = consultRepository.findById(consultId)
			.orElseThrow(() -> new BaseExceptionHandler(NOT_FOUND_CONSULT_EXCEPTION));

		//channelId가 없는 경우 예외 처리
		String channelId = consult.getChannelId();
		if (channelId == null || channelId.isEmpty()) {
			throw new BaseExceptionHandler(CHANNEL_NOT_FOUND_EXCEPTION);
		}

		//고민상담소가 닫혀있는지 확인 -> 이미 닫힌 고민상담소는 나가기 불가
		if (consult.isClosed()) {
			throw new BaseExceptionHandler(ALREADY_CLOSED_EXCEPTION);
		}

		//나가려는 사람이 상담소를 연 사람인지 확인 -> 그렇다면 예외 처리
		//상담소를 연 사람은 상담 자체를 종료하거나 참여자를 내보내기만 할 수 있음
		if (consult.getMemberId() == sender) {
			throw new BaseExceptionHandler(UNAUTHORIZED_USER_EXCEPTION);
		}

		//channelId가 있다면 삭제
		consult.setChannelId(null);

		// TODO 알림 발생 : notificationms에 전송
//        notificationClient.consultNotification("EXIT", consult.getMemberId());
	}

	//참여자 강제로 추방시키기
	@Override
	public void expelConsultingRoom(int consultId, int sender) throws BaseExceptionHandler {
		//consultId 기반으로 현재 Consult 조회
		Consult consult = consultRepository.findById(consultId)
			.orElseThrow(() -> new BaseExceptionHandler(NOT_FOUND_CONSULT_EXCEPTION));

		//channelId가 없는 경우 예외 처리
		String channelId = consult.getChannelId();
		if (channelId == null || channelId.isEmpty()) {
			throw new BaseExceptionHandler(CHANNEL_NOT_FOUND_EXCEPTION);
		}

		//고민상담소가 닫혀있는지 확인 -> 이미 닫힌 고민상담소는 퇴장 불가
		if (consult.isClosed()) {
			throw new BaseExceptionHandler(ALREADY_CLOSED_EXCEPTION);
		}

		//퇴장시키려는 사람이 상담소를 열지 않은 사람인지 확인 -> 그렇다면 예외 처리
		if (consult.getMemberId() != sender) {
			throw new BaseExceptionHandler(UNAUTHORIZED_USER_EXCEPTION);
		}

		//channelId가 있다면 삭제
		consult.setChannelId(null);

		// TODO 알림 발생 : notificationms에 전송
//        Channel channel = channelRepository.findById(channelId).get();
//        notificationClient.consultNotification("BANNED",
//            Integer.parseInt(channel.getReceiver().get("memberId")));

	}

	//대화중인 채팅방 목록(나의 고민)
	@Override
	public ConsultChattingListRes getMyChattingRooms(int memberId) {

		//내가 참여중인 consult 정보 가져오기
		List<Consult> consultList = consultRepository.findAllByMemberIdOrderByUpdateTimeDesc(
			memberId);

		// 없으면 null 출력
		if (consultList == null || consultList.isEmpty()) {
			return null;
		}

		// channelId가 null이 아닌 값들만 필터링
		List<Consult> filteredConsultList = consultList.stream()
			.filter(consult -> consult.getChannelId() != null).collect(Collectors.toList());

		// ConsultChattingRes 객체의 리스트를 저장할 변수 선언
		List<ConsultChattingRes> consultChattingResList = new ArrayList<>();

		//channel에 대한 정보 가져오기 -> 없으면 에러처리
		for (Consult consult : filteredConsultList) {
			//채널 정보 가져오기
			Channel channel = channelRepository.findById(consult.getChannelId()).get();

			if (!channel.getMessageList().isEmpty()) {
				//가장 최근 메세지 한개 가져오기
				Message latestMessage = channel.getMessageList()
					.get(channel.getMessageList().size() - 1);

				// ConsultChattingRes 객체 생성 및 최근 메시지 설정
				ConsultChattingRes consultChattingRes = ConsultChattingRes.builder()
					.consultId(consult.getConsultId()).memberId(consult.getMemberId())
					.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
					.title(consult.getTitle()).channelId(consult.getChannelId())
					.text(latestMessage.getText()) // 최근 메시지의 텍스트 설정
					.shared(consult.isShared())
					.build();

				// ConsultChattingRes 객체를 리스트에 추가
				consultChattingResList.add(consultChattingRes);
			} else {
				// ConsultChattingRes 객체 생성 및 최근 메시지 설정
				ConsultChattingRes consultChattingRes = ConsultChattingRes.builder()
					.consultId(consult.getConsultId()).memberId(consult.getMemberId())
					.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
					.title(consult.getTitle()).channelId(consult.getChannelId())
					.text(null) // 최근 메시지의 텍스트 설정
					.shared(consult.isShared())
					.build();

				// ConsultChattingRes 객체를 리스트에 추가
				consultChattingResList.add(consultChattingRes);
			}
		}

		return ConsultChattingListRes.builder()
			.consultChattingRes(consultChattingResList)
			.build();
	}

	//대화중인 채팅방 목록(다른사람 고민)
	@Override
	public ConsultChattingListRes getOthersChattingRooms(int memberId) {

		// ConsultChattingRes 객체의 리스트를 저장할 변수 선언
		List<ConsultChattingRes> consultChattingResList = new ArrayList<>();

		//내가 참여중인 채널의 정보 가져오기
		List<Channel> channelList = channelRepository.findBySender(String.valueOf(memberId));

//        log.error("[channelList] channelList={}",channelList.get(0).getChannelId());

		if (channelList.isEmpty()) {
			return null;
		}

		for (Channel c : channelList) {

			Consult consult = consultRepository.findByChannelId(c.getChannelId());

			log.debug("[getOthersChattingRooms] channelId={}", c.getChannelId());
			if (consult == null) {
				continue;
			}

			//가장 최근 메세지 한개 가져오기
			Message latestMessage = null;
			if (!c.getMessageList().isEmpty()) {
				latestMessage = c.getMessageList().get(c.getMessageList().size() - 1);
			}

			String text = null;
			if (latestMessage != null) {
				text = latestMessage.getText();
			}

			// ConsultChattingRes 객체 생성 및 최근 메시지 설정
			ConsultChattingRes consultChattingRes = ConsultChattingRes.builder()
				.consultId(consult.getConsultId()).memberId(consult.getMemberId())
				.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
				.title(consult.getTitle()).channelId(consult.getChannelId())
				.text(text) // 최근 메시지의 텍스트 설정
				.shared(consult.isShared())
				.build();

			// ConsultChattingRes 객체를 리스트에 추가
			consultChattingResList.add(consultChattingRes);
		}

		return ConsultChattingListRes.builder()
			.consultChattingRes(consultChattingResList)
			.build();
	}

	//공유된 고민상담 내용에 좋아요 등록
	@Override
	@Transactional
	public void addLikesToRedis(int consultId, int memberId) throws BaseExceptionHandler {

		HashOperations<String, String, String> hashOperations = redisTemplate.opsForHash();
		String key = "likeConsultId::" + String.valueOf(consultId);
		String field = String.valueOf(memberId);

		if (hashOperations.get(key, field) == null) {
			LikeConsult likeConsult = likeConsultRepository.findByConsultIdAndMemberId(consultId,
				memberId);

			//이미 좋아요 등록이 되어 있는 경우
			if (likeConsult != null) {
				throw new BaseExceptionHandler(ALREADY_CONSULT_LIKE_EXISTS);
			}
			hashOperations.put(key, field, String.valueOf(1));

		} else {
			throw new BaseExceptionHandler(ALREADY_CONSULT_LIKE_EXISTS);
		}

		log.debug("postLikeConsults method consultId: {} memberId:{} success ", consultId,
			memberId);
	}

	//공유된 고민상담 내용에 좋아요 등록 취소
	@Override
	@Transactional
	public void deleteLikePostit(int consultId, int memberId) {

		HashOperations<String, String, String> hashOperations = redisTemplate.opsForHash();
		String key = "likeConsultId::" + String.valueOf(consultId);
		String field = String.valueOf(memberId);

		if (hashOperations.get(key, field) == null) {
			LikeConsult likeConsult = likeConsultRepository.findByConsultIdAndMemberId(consultId,
				memberId);

			//좋아요가 등록이 안되어 있는 경우
			if (likeConsult == null) {
				throw new BaseExceptionHandler(NOT_FOUND_CONSULT_LIKE_EXISTS);
			} else {
				likeConsultRepository.delete(likeConsult);
				Consult consult = consultRepository.findById(consultId).get();
				int likeCount = consult.getLikeCount() - 1;
				consult.setLikeCount(Math.max(likeCount, 0));
				consultRepository.save(consult);
			}
		} else {
			hashOperations.delete(key, field);
		}

		log.debug("deleteLikeConsults method consultId: {} memberId:{} success ", consultId,
			memberId);
	}

	//카테고리로 대화가능한 고민상담소 필터링
	@Override
	public ConsultListRes getConsultListByCategory(int categoryId) {

		List<Consult> consultList = new ArrayList<>();

		//카테고리ID가 1이면 전체 값 조회
		if (categoryId == 1) {
			//consultRepository에 저장되어 있는 모든 값 가져오기
			consultList = consultRepository.findAllByIsClosedAndChannelIdOrderByCreateTimeDesc(
				false, null);
		} else {
			//카테고리 ID가 그 외의 값이면 해당하는 값 조회
			consultList = consultRepository.findAllByCategoryIdAndIsClosedAndChannelIdOrderByUpdateTimeDesc(
				categoryId, false, null);
		}

		// Consult 객체의 리스트를 ConsultDetailRes 객체의 리스트로 변환
		List<ConsultDetailRes> consultDetailResList = consultList.stream().map(
			consult -> ConsultDetailRes.builder().consultId(consult.getConsultId())
				.memberId(consult.getMemberId())
				.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
				.title(consult.getTitle()).content(consult.getContent())
				.categoryId(consult.getCategoryId()).isClosed(consult.isClosed())
				.isShared(consult.isShared()).canLike(consult.isCanLike())
				.channelId(consult.getChannelId()).build()).collect(Collectors.toList());

		return ConsultListRes.builder().consultList(consultDetailResList)
			.build();

	}

	//카테고리로 공유된 고민상담소 필터링
	@Override
	public ConsultListRes getSharedConsultListByCategory(int categoryId) {

		List<Consult> consultList = new ArrayList<>();

		//카테고리ID가 1이면 전체 값 조회
		if (categoryId == 1) {
			//consultRepository에 저장되어 있는 값들 중 공유된 고민상담소 가져오기
			consultList = consultRepository.findAllByIsSharedOrderByCreateTimeDesc(true);
		} else {
			//카테고리 ID가 그 외의 값이면 해당하는 값 조회
			consultList = consultRepository.findAllByCategoryIdAndIsSharedOrderByUpdateTimeDesc(
				categoryId, true);
		}

		// Consult 객체의 리스트를 ConsultDetailRes 객체의 리스트로 변환
		List<ConsultDetailRes> consultDetailResList = consultList.stream().map(
			consult -> ConsultDetailRes.builder().consultId(consult.getConsultId())
				.memberId(consult.getMemberId())
				.nickname(memberClient.getMember(consult.getMemberId()).getResult().nickname())
				.title(consult.getTitle()).content(consult.getContent())
				.categoryId(consult.getCategoryId()).isClosed(consult.isClosed())
				.isShared(consult.isShared()).canLike(consult.isCanLike())
				.likeCount(consult.getLikeCount())
				.channelId(consult.getChannelId()).build()).collect(Collectors.toList());

		return ConsultListRes.builder().consultList(consultDetailResList)
			.build();

	}

	@Override
	public void makeNotification(String type, int memberId) {
		// 알림 전송 kafka(notification에서는 알림테이블에 저장 + 실시간 알림 전송)

		NotificationEventDto eventDto = NotificationEventDto.builder()
			.eventType("Consult-" + type)
			.memberId(memberId)
			.build();

		ObjectMapper objectMapper = new ObjectMapper();
		String jsonString;
		// 객체를 JSON 문자열로 변환
		try {
			jsonString = objectMapper.writeValueAsString(eventDto);
			notificationEventDtoKafkaTemplate.send("notification-topic", jsonString);

		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

//		notificationClient.consultNotification(type, memberId);
	}

	private boolean isLike(int consultId, int memberId) {

		HashOperations<String, String, String> hashOperations = redisTemplate.opsForHash();

		String key = "likeConsultId::" + String.valueOf(consultId);
		String field = String.valueOf(memberId);

		if (hashOperations.get(key, field) == null) {
			return likeConsultRepository.findByConsultIdAndMemberId(consultId, memberId) != null;
		} else {
			return true;
		}

	}
}
