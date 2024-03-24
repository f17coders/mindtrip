package com.a303.memberms.domain.member.controller;

import com.a303.memberms.domain.member.dto.request.MemberStandardLoginReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardRegisterReq;
import com.a303.memberms.domain.member.dto.response.MemberBaseRes;
import com.a303.memberms.domain.member.service.MemberService;
import com.a303.memberms.global.api.response.BaseResponse;
import com.a303.memberms.global.exception.code.SuccessCode;
import io.micrometer.core.annotation.Timed;
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
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.a303.memberms.domain.member.dto.request.MemberStandardLoginReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardRegisterReq;
import com.a303.memberms.domain.member.dto.response.MemberBaseRes;
import com.a303.memberms.domain.member.service.MemberService;
import com.a303.memberms.global.api.response.BaseResponse;
import com.a303.memberms.global.exception.code.SuccessCode;

import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@RestController
@RequestMapping("/api/members")
@RequiredArgsConstructor
@CrossOrigin("*")
@Slf4j
public class MemberController {

	private final MemberService memberService;

	//    @Operation(summary = "health check")
	@GetMapping("/welcome")
	public ResponseEntity<BaseResponse<String>> welcome() throws IOException {

		log.info("Entering the member-controller  ");

		return BaseResponse.success(SuccessCode.CHECK_SUCCESS, "its memberms");
	}


	//------------------------- 다른 msa와 통신 -------------------------------
	//    @Operation(summary = "멤버아이디로 멤버 조회")
	@GetMapping("/v1/{memberId}")
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

	@PostMapping("/v1/login")
	public ResponseEntity<BaseResponse<String>> login(
		@RequestBody
		MemberStandardLoginReq memberStandardLoginReq,
		HttpServletResponse response
	) {
		String token = memberService.standardLogin(memberStandardLoginReq);
		response.setHeader("Authorization", "Bearer " + token);

		log.debug("Authorization: {}", token);

		return BaseResponse.success(
			SuccessCode.LOGIN_SUCCESS,
			"로그인 성공"
		);
	}

	@PostMapping("/v1/register")
	public ResponseEntity<BaseResponse<String>> register(
		@RequestBody
		MemberStandardRegisterReq memberStandardRegisterReq
	) {
		memberService.standardRegister(memberStandardRegisterReq);

		return BaseResponse.success(
			SuccessCode.INSERT_SUCCESS,
			"회원가입 성공"
		);
	}

	@GetMapping("/v1/availability/id")
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

	@GetMapping("/v1/availability/nickname")
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
}

