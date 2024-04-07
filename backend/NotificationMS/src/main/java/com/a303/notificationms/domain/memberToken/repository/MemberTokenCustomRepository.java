package com.a303.notificationms.domain.memberToken.repository;

import com.a303.notificationms.domain.memberToken.MemberToken;

public interface MemberTokenCustomRepository {

    MemberToken findByMemberId(int memberId);

}
