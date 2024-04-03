package com.a303.memberms.domain.member.service;

import com.a303.memberms.domain.member.Member;
import com.a303.memberms.domain.member.dto.request.AuthTokenReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardLoginReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardRegisterReq;
import com.a303.memberms.domain.member.dto.response.MemberBaseRes;

import com.a303.memberms.domain.member.dto.response.MemberLoginRes;
import com.a303.memberms.domain.member.repository.MemberRepository;
import com.a303.memberms.domain.notification.dto.response.NotificationEventDto;
import com.a303.memberms.domain.village.dto.response.VillageBaseRes;
import com.a303.memberms.global.api.response.BaseResponse;

import com.a303.memberms.global.client.AuthClient;
import com.a303.memberms.global.client.VillageClient;

import com.a303.memberms.global.exception.BaseExceptionHandler;
import com.a303.memberms.global.exception.code.ErrorCode;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;


@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class MemberServiceImpl implements MemberService {

	private final MemberRepository memberRepository;
	private final PasswordEncoder passwordEncoder;

	private final AuthClient authClient;
	private final VillageClient villageClient;



	@Override
	public MemberBaseRes getMemberByMemberId(int memberId)
		throws BaseExceptionHandler, IOException {
		Member member = memberRepository.findByMemberId(memberId).orElseThrow(
			() -> new BaseExceptionHandler(
				ErrorCode.NOT_FOUND_USER_EXCEPTION
			)
		);

		VillageBaseRes villageBaseRes;
		if (member.getVillageId() == null) {
			villageBaseRes = VillageBaseRes.builder().build();
		} else {
			villageBaseRes = villageClient.getVillage(member.getVillageId()).getResult();
		}

		return MemberBaseRes.builder()
			.memberId(memberId)
			.id(member.getId())
			.password(member.getPassword())
			.socialId(member.getSocialId())
			.nickname(member.getNickname())
			.villageId(villageBaseRes.villageId())
			.villageName(villageBaseRes.villageName())
			.level(member.getLevel())
			.missionCount(member.getMissionCount())
			.reportCount(member.getReportCount())
			.role(member.getRole())
			.build();
	}


	@Override
	public List<Integer> getMemberIdList() throws BaseExceptionHandler, IOException {
		List<Integer> memberIdList = memberRepository.findDistinctMemberId();

		return memberIdList;
	}


	@Override
	public MemberLoginRes standardLogin(MemberStandardLoginReq memberStandardLoginReq) {
		//ID로 존재 여부 확인(사실상 일치 여부도 확인됨)
		Member target = memberRepository.findById(memberStandardLoginReq.id())
			.orElseThrow(
				() -> new BaseExceptionHandler(
					"아이디를 확인해주세요.",
					ErrorCode.NOT_FOUND_USER_EXCEPTION
				)
			);

		//패스워드 일치 확인
		if (!passwordEncoder.matches(
			memberStandardLoginReq.password(),
			target.getPassword()
		)) {
			throw new BaseExceptionHandler(
				"패스워드를 확인해주세요.",
				ErrorCode.NOT_FOUND_USER_EXCEPTION
			);
		}

		//FeignClient로 토큰 발급
		ResponseEntity<BaseResponse<String>> response = authClient.token(
			AuthTokenReq.builder()
				.memberId(target.getMemberId())
				.role(target.getRole().name())
				.build()
		);
		String token = response.getBody().getResult();

		return MemberLoginRes.builder()
			.memberId(target.getMemberId())
			.token("Bearer " + token)
			.build();

	}


	@Override
	@Transactional
	public int standardRegister(MemberStandardRegisterReq memberStandardRegisterReq) {
		//ID 중복 체크
		String id = memberStandardRegisterReq.id();
		checkIdDuplication(id);

		//닉네임 중복 체크
		String nickname = memberStandardRegisterReq.nickname();
		checkNicknameDuplication(nickname);

		//패스워드 인코딩 + DB에 넣기
		Member member = Member.createMember(
			memberStandardRegisterReq.id(),
			passwordEncoder.encode(memberStandardRegisterReq.password()),
			memberStandardRegisterReq.nickname()
		);
		Member registeredMember = memberRepository.save(member);

		return registeredMember.getMemberId();
	}


	public void checkIdDuplication(String id) {
		if (memberRepository.existsById(id)) {
			throw new BaseExceptionHandler(
				"이미 존재하는 아이디입니다.",
				ErrorCode.ID_ALREADY_EXISTS_EXCEPTION
			);
		}
	}

	public void checkNicknameDuplication(String nickname) {
		if (memberRepository.existsByNickname(nickname)) {
			throw new BaseExceptionHandler(
				"이미 존재하는 닉네임입니다.",
				ErrorCode.NICKNAME_ALREADY_EXISTS_EXCEPTION
			);
		}
	}


	@Override
	public void increaseMissionCountByMemberId(int memberId) {

		Member member = memberRepository.findByMemberId(memberId).orElseThrow(
			() -> new BaseExceptionHandler(
				ErrorCode.NOT_FOUND_USER_EXCEPTION
			)
		);

		int newMissionCount = member.getMissionCount() + 1;
		member.setMissionCount(newMissionCount);
		// 만약 임계치 넘었다면 레벨 업 시켜야함.
		memberRepository.increaseLevel(member, newMissionCount);
	}


	@Override
	public int getMissionCount(int memberId) {
		return memberRepository.findMissionCountByMemberId(memberId);
	}

	@Override
	public MemberBaseRes changeNickname(int memberId, String nickname) {

		Member member = memberRepository.findByMemberId(memberId).orElseThrow(
			() -> new BaseExceptionHandler(
				ErrorCode.NOT_FOUND_USER_EXCEPTION
			)
		);

		member.setNickname(nickname);

		VillageBaseRes villageBaseRes;
		if (member.getVillageId() == null) {
			villageBaseRes = VillageBaseRes.builder().build();
		} else {
			villageBaseRes = villageClient.getVillage(member.getVillageId()).getResult();
		}

		return MemberBaseRes.builder()
			.memberId(memberId)
			.id(member.getId())
			.password(member.getPassword())
			.socialId(member.getSocialId())
			.nickname(member.getNickname())
			.villageId(villageBaseRes.villageId())
			.villageName(villageBaseRes.villageName())
			.level(member.getLevel())
			.missionCount(member.getMissionCount())
			.reportCount(member.getReportCount())
			.role(member.getRole())
			.build();
	}


}
