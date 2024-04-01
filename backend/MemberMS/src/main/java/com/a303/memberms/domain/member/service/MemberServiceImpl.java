package com.a303.memberms.domain.member.service;

import com.a303.memberms.domain.member.Member;
import com.a303.memberms.domain.member.dto.request.AuthTokenReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardLoginReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardRegisterReq;
import com.a303.memberms.domain.member.dto.response.MemberBaseRes;
import com.a303.memberms.domain.member.repository.MemberRepository;
import com.a303.memberms.global.api.response.BaseResponse;
import com.a303.memberms.global.exception.BaseExceptionHandler;
import com.a303.memberms.global.exception.code.ErrorCode;
import com.a303.memberms.global.client.AuthClient;

import jakarta.transaction.Transactional;
import java.io.IOException;
import java.util.List;
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

	@Override
	public MemberBaseRes getMemberByMemberId(int memberId)
		throws BaseExceptionHandler, IOException {

		Member member = memberRepository.findByMemberId(memberId).orElseThrow(
			() -> new BaseExceptionHandler(
				ErrorCode.NOT_FOUND_USER_EXCEPTION
			)
		);

		return MemberBaseRes.builder()
			.memberId(memberId)
			.id(member.getId())
			.password(member.getPassword())
			.socialId(member.getSocialId())
			.nickname(member.getNickname())
			.villageId(member.getVillageId())
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
    public String standardLogin(MemberStandardLoginReq memberStandardLoginReq) {
        //1. ID로 존재 여부 확인(사실상 일치 여부도 확인됨)
        Member target = memberRepository.findById(memberStandardLoginReq.id())
            .orElseThrow(
                () -> new BaseExceptionHandler(
                    "ID 없어잉",
                    ErrorCode.NOT_FOUND_USER_EXCEPTION
                )
            );

        //2. 패스워드 일치 확인
//        if (!passwordEncoder.matches(
//            memberStandardLoginReq.password(),
//            target.getPassword()
//        )) {
        if(!memberStandardLoginReq.password().equals(target.getPassword())) {
            throw new BaseExceptionHandler(
                "PW 이상행",
                ErrorCode.NOT_FOUND_USER_EXCEPTION
            );
        }

        //3. FeignClient로 토큰 발급
        ResponseEntity<BaseResponse<String>> response = authClient.token(
            AuthTokenReq.builder()
                .memberId(target.getMemberId())
                .role(target.getRole().name())
                .build()
        );

        return response.getBody().getResult();
    }

    @Override
    @Transactional
    public int standardRegister(MemberStandardRegisterReq memberStandardRegisterReq) {
        //1. ID 형식 체크(MVP에서는 안함)
        //2. ID 중복 체크
        //3. 닉네임 형식 체크(MVP에서는 안함)
        //4. 닉네임 중복 체크
        //5. 패스워드 형식 체크(MVP에서는 안함)
        //6. 패스워드 인코딩(MVP에서는 일단 안함)
        //7. DB에 넣기

        //2. ID 중복 체크
		String id = memberStandardRegisterReq.id();
        checkIdDuplication(id);

        //4. 닉네임 중복 체크
        String nickname = memberStandardRegisterReq.nickname();
        checkNicknameDuplication(nickname);

        //6. 패스워드 인코딩
		//...은 나중에

		//7. DB에 넣기
        Member member = Member.createMember(
			memberStandardRegisterReq.id(),
			memberStandardRegisterReq.password(),
			memberStandardRegisterReq.nickname()
		);
		Member registered = memberRepository.save(member);

		return registered.getMemberId();
	}

	public void checkIdDuplication(String id) {
        if(memberRepository.existsById(id)) {
			throw new BaseExceptionHandler(
				"이미 존재하는 아이디입니다.",
				ErrorCode.ID_ALREADY_EXISTS_EXCEPTION
			);
		}
    }

    public void checkNicknameDuplication(String nickname) {
        if(memberRepository.existsByNickname(nickname)) {
			throw new BaseExceptionHandler(
				"이미 존재하는 닉네임입니다.",
				ErrorCode.NICKNAME_ALREADY_EXISTS_EXCEPTION
			);
		}
    }

}

