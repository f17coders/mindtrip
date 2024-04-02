package com.a303.memberms.domain.member.controller;

import com.a303.memberms.domain.member.dto.request.MemberStandardLoginReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardRegisterReq;
import com.a303.memberms.domain.member.dto.response.MemberBaseRes;
import com.a303.memberms.domain.member.dto.response.MemberLoginRes;
import com.a303.memberms.domain.member.service.MemberService;
import com.a303.memberms.global.api.response.BaseResponse;
import com.a303.memberms.global.exception.code.SuccessCode;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/members")
@RequiredArgsConstructor
@CrossOrigin("*")
@Slf4j
public class MemberController {

	private final MemberService memberService;

	//    @Operation(summary = "health check")

	@GetMapping("/v1/mission-count")
	public ResponseEntity<BaseResponse<Integer>> getMissionCount(
		@RequestHeader("x-member-id") int memberId
	) throws IOException {

		log.debug("members/v1/mission-count GET api accepted with memberId:{}", memberId);

		int count = memberService.getMissionCount(memberId);

		log.debug("members/v1/mission-count GET api successed with memberId:{}", memberId);

		return BaseResponse.success(SuccessCode.CHECK_SUCCESS, count);
	}

	@PutMapping("/v1/nickname")
	public ResponseEntity<BaseResponse<MemberBaseRes>> changeNickname(
		@RequestHeader("x-member-id") int memberId,
		@RequestParam("nickname") String nickname
	) throws IOException {

		log.debug("members/v1/nickname PUT api accepted with memberId:{}", memberId);

		MemberBaseRes memberBaseRes = memberService.changeNickname(memberId, nickname);

		log.debug("members/v1/nickname PUT api successed with memberId:{}", memberId);

		return BaseResponse.success(SuccessCode.CHECK_SUCCESS, memberBaseRes);
	}


	//------------------------- 다른 msa와 통신 -------------------------------
	//    @Operation(summary = "멤버아이디로 멤버 조회")
	@GetMapping("/v0/{memberId}")
	public ResponseEntity<BaseResponse<MemberBaseRes>> getMemberDtoByMemberId(
		@PathVariable("memberId") int memberId
	) throws IOException {

		MemberBaseRes memberBaseRes = memberService.getMemberByMemberId(memberId);

		return BaseResponse.success(SuccessCode.CHECK_SUCCESS, memberBaseRes);
	}

	@GetMapping("/v0/id-list")
	public ResponseEntity<BaseResponse<List<Integer>>> getMemberIdList() throws IOException {

		List<Integer> memberIdList = memberService.getMemberIdList();

		return BaseResponse.success(SuccessCode.CHECK_SUCCESS, memberIdList);
	}

	@PostMapping("/v0/login")
	public ResponseEntity<BaseResponse<MemberLoginRes>> login(
		@RequestBody
		MemberStandardLoginReq memberStandardLoginReq,
		HttpServletResponse response
	) {
        MemberLoginRes memberLoginRes = memberService.standardLogin(memberStandardLoginReq);
//		response.setHeader("Authorization", "Bearer " + token);

		log.debug("Authorization: {}", memberLoginRes);
		log.info("{} login", memberStandardLoginReq.id());

		return BaseResponse.success(
			SuccessCode.LOGIN_SUCCESS,
            memberLoginRes
		);
	}

	@PostMapping("/v0/register")
	public ResponseEntity<BaseResponse<Integer>> register(
		@RequestBody
		MemberStandardRegisterReq memberStandardRegisterReq
	) {
		int registeredMemberId = memberService.standardRegister(memberStandardRegisterReq);

		return BaseResponse.success(
			SuccessCode.INSERT_SUCCESS,
			registeredMemberId
		);
	}

	@GetMapping("/v0/availability/id")
	public ResponseEntity<BaseResponse<String>> checkIdAvailability(
		@RequestParam
		String id
	) {
		memberService.checkIdDuplication(id);

		return BaseResponse.success(
			SuccessCode.AVAILABLE_ID,
			"사용 가능한 아이디입니다."
		);
	}


	@GetMapping("/v0/availability/nickname")
	public ResponseEntity<BaseResponse<String>> checkNicknameAvailability(
		@RequestParam
		String nickname
	) {
		memberService.checkNicknameDuplication(nickname);

		return BaseResponse.success(
			SuccessCode.AVAILABLE_NICKNAME,
			"사용 가능한 닉네임입니다."
		);
	}


	@PutMapping("/v0/mission-count")
	public ResponseEntity<BaseResponse<String>> increaseMissionCount(
		@RequestParam("memberId") int memberId

	) {
		memberService.increaseMissionCountByMemberId(memberId);

		return BaseResponse.success(
			SuccessCode.UPDATE_SUCCESS,
			"증가 완료"
		);
	}

}

