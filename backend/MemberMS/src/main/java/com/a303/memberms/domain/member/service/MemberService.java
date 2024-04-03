package com.a303.memberms.domain.member.service;

import com.a303.memberms.domain.member.dto.request.MemberStandardLoginReq;
import com.a303.memberms.domain.member.dto.request.MemberStandardRegisterReq;
import com.a303.memberms.domain.member.dto.response.MemberBaseRes;
import com.a303.memberms.domain.member.dto.response.MemberLoginRes;
import com.a303.memberms.global.exception.BaseExceptionHandler;
import java.io.IOException;
import java.util.List;

public interface MemberService {

	MemberBaseRes getMemberByMemberId(int memberId)
		throws BaseExceptionHandler, IOException;

	List<Integer> getMemberIdList()
		throws BaseExceptionHandler, IOException;

    MemberLoginRes standardLogin(MemberStandardLoginReq memberStandardLoginReq);

    int standardRegister(MemberStandardRegisterReq memberStandardRegisterReq);

	void checkIdDuplication(String id);
	void checkNicknameDuplication(String nickname);

	void increaseMissionCountByMemberId(int memberId);

	int getMissionCount(int memberId);

	MemberBaseRes changeNickname(int memberId, String nickname);


}
